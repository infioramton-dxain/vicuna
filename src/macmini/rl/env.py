""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for the RL env; install into dl_venv: "
        "`dl_venv/bin/pip install gymnasium stable-baselines3 sb3-contrib`"
    ) from exc

log = logging.getLogger(__name__)


K_SLOTS = 25
EPISODE_LEN = 63
INIT_CASH = 100_000.0
POSITION_SIZE_FRAC = 0.05
MAX_POSITIONS = K_SLOTS
DD_TERMINATE = -0.20
DD_THRESHOLD = -0.10
HHI_THRESHOLD = 0.30
LAMBDA_VOL = 0.0
LAMBDA_DD = 2.0
LAMBDA_CONC = 1.0
FEE_BPS = 8.0
BETA_TERM = 0.02
DL_STALENESS_DAYS = 7

OBS_DIM_PORTFOLIO = 7
OBS_DIM_REGIME = 5

ACTION_DIM = 1 + 2 * K_SLOTS

SECTOR_ORDER = ("Materials", "Mining", "Defense", "Software", "Computers", "Other")

OBS_DIM_PER_SLOT = 8 + len(SECTOR_ORDER) + 1
OBS_DIM = OBS_DIM_PORTFOLIO + OBS_DIM_REGIME + K_SLOTS * OBS_DIM_PER_SLOT


@dataclass
class Position:
    shares: float = 0.0
    avg_cost: float = 0.0
    entry_step: int = -1


@dataclass
class PortfolioState:
    cash: float = INIT_CASH
    positions: dict[str, Position] = field(default_factory=dict)
    port_value_series: list[float] = field(default_factory=lambda: [INIT_CASH])
    step_returns: list[float] = field(default_factory=list)
    running_max: float = INIT_CASH
    prev_action: int = 0
    prev_dd_excess: float = 0.0
    prev_hhi_excess: float = 0.0

    def port_value(self, prices: dict[str, float]) -> float:
        v = self.cash
        for sym, p in self.positions.items():
            if p.shares > 0:
                v += p.shares * prices.get(sym, p.avg_cost)
        return v


class UniverseData:
    ""

    def __init__(
        self,
        db_path: Path,
        universe_symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        self.db_path = Path(db_path)
        self.symbols = list(universe_symbols)
        self.symbol_to_slot = {s: i for i, s in enumerate(self.symbols)}
        self.start_date = start_date
        self.end_date = end_date
        self._load()

    def _load(self) -> None:
        placeholders = ",".join("?" for _ in self.symbols)
        with sqlite3.connect(self.db_path, timeout=30) as c:
            self.bars = pd.read_sql_query(
                f"""SELECT symbol, date, open, high, low, close, volume
                    FROM mm_bars
                    WHERE symbol IN ({placeholders})
                      AND date BETWEEN ? AND ?
                    ORDER BY symbol, date""",
                c, params=[*self.symbols, self.start_date, self.end_date],
            )
            self.indicators = pd.read_sql_query(
                f"""SELECT symbol, date, rsi14, macd_hist, atr14, volume_zscore_50
                    FROM mm_indicators_daily
                    WHERE symbol IN ({placeholders})
                      AND date BETWEEN ? AND ?""",
                c, params=[*self.symbols, self.start_date, self.end_date],
            )
            self.dl = pd.read_sql_query(
                f"""SELECT symbol, date, predicted_5d_return
                    FROM mm_dl_signals
                    WHERE symbol IN ({placeholders})
                      AND date BETWEEN ? AND ?""",
                c, params=[*self.symbols, self.start_date, self.end_date],
            )
            self.regime = pd.read_sql_query(
                """SELECT as_of_date AS date, spy_50d_above_200d, vix_close,
                          vix_regime, fomc_within_3d
                   FROM mm_regime
                   WHERE as_of_date BETWEEN ? AND ?""",
                c, params=[self.start_date, self.end_date],
            )
            self.sector_map = pd.read_sql_query(
                f"SELECT symbol, sector FROM mm_sector_map "
                f"WHERE symbol IN ({placeholders})",
                c, params=self.symbols,
            )

        self.trading_dates: list[str] = sorted(
            self.bars.loc[self.bars["symbol"] == "SPY", "date"].astype(str).unique().tolist()
        )
        if not self.trading_dates:
            self.trading_dates = sorted(self.bars["date"].astype(str).unique().tolist())

        self.bars["date_str"] = self.bars["date"].astype(str).str.slice(0, 10)
        self.indicators["date_str"] = self.indicators["date"].astype(str).str.slice(0, 10)
        self.dl["date_str"] = self.dl["date"].astype(str).str.slice(0, 10)
        self.regime["date_str"] = self.regime["date"].astype(str).str.slice(0, 10)
        self._close_by_sd = {(r.symbol, r.date_str): r.close for r in self.bars.itertuples()}
        self._open_by_sd = {(r.symbol, r.date_str): r.open for r in self.bars.itertuples()}
        self._ind_by_sd = {
            (r.symbol, r.date_str): (r.rsi14, r.macd_hist, r.atr14, r.volume_zscore_50)
            for r in self.indicators.itertuples()
        }
        self._dl_by_sd = {
            (r.symbol, r.date_str): r.predicted_5d_return
            for r in self.dl.itertuples() if not pd.isna(r.predicted_5d_return)
        }
        self._regime_by_d = {
            r.date_str: (r.spy_50d_above_200d, r.vix_close, r.vix_regime, r.fomc_within_3d)
            for r in self.regime.itertuples()
        }
        self._sector_by_sym = dict(zip(self.sector_map["symbol"], self.sector_map["sector"]))

    def open_price(self, symbol: str, date_str: str) -> float | None:
        return self._open_by_sd.get((symbol, date_str))

    def close_price(self, symbol: str, date_str: str) -> float | None:
        return self._close_by_sd.get((symbol, date_str))

    def indicators_row(self, symbol: str, date_str: str) -> tuple[float, ...] | None:
        return self._ind_by_sd.get((symbol, date_str))

    def dl_prediction(self, symbol: str, date_str: str) -> float | None:
        ""
        v = self._dl_by_sd.get((symbol, date_str))
        if v is not None:
            return float(v)
        d0 = pd.Timestamp(date_str)
        for back in range(1, DL_STALENESS_DAYS + 1):
            dprev = (d0 - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            v = self._dl_by_sd.get((symbol, dprev))
            if v is not None:
                return float(v)
        return None

    def regime_row(self, date_str: str) -> tuple | None:
        return self._regime_by_d.get(date_str)

    def sector(self, symbol: str) -> str:
        return self._sector_by_sym.get(symbol, "Other")


class MacminiSwingEnv(gym.Env):
    ""

    metadata = {"render_modes": []}

    def __init__(
        self,
        db_path: Path | str,
        universe_topk_parquet: Path | str | None = None,
        universe_symbols: Iterable[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2025-06-30",
        episode_len: int = EPISODE_LEN,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.db_path = Path(db_path)

        default_cache = Path(__file__).resolve().parents[3] / "data/rl_cache/universe_topk.parquet"
        cache_path = Path(universe_topk_parquet) if universe_topk_parquet is not None else default_cache
        self._universe_by_date: dict[str, list[str]] = {}
        if cache_path.exists() and universe_symbols is None:
            self._load_universe_cache(cache_path)
            universe_symbols_effective = sorted({s for lst in self._universe_by_date.values() for s in lst})
            log.info("rl env universe: dynamic per-date, %d unique symbols, %d dates covered",
                     len(universe_symbols_effective), len(self._universe_by_date))
        else:
            log.info("rl env universe: FIXED (no universe cache found or explicit symbols passed)")
            if universe_symbols is None:
                universe_symbols = self._read_default_universe()
            universe_symbols = list(universe_symbols)[:K_SLOTS]
            universe_symbols_effective = universe_symbols

        symbols_to_load = list({*universe_symbols_effective, "SPY"})

        self.data = UniverseData(self.db_path, symbols_to_load, start_date, end_date)
        self.trading_dates = self.data.trading_dates
        if len(self.trading_dates) < episode_len + 5:
            raise ValueError(
                f"universe has only {len(self.trading_dates)} trading days in-window; "
                f"need at least {episode_len + 5} for an episode"
            )
        self.episode_len = episode_len

        self._fixed_universe: list[str] | None = None
        if not self._universe_by_date:
            fu = list(universe_symbols_effective)[:K_SLOTS]
            if len(fu) < K_SLOTS:
                fu += ["__PAD__"] * (K_SLOTS - len(fu))
            self._fixed_universe = fu

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _load_universe_cache(self, path: Path) -> None:
        df = pd.read_parquet(path)
        df["date_str"] = df["date"].astype(str).str.slice(0, 10)
        df = df.sort_values(["date_str", "rank"])
        self._universe_by_date = {
            d: g["symbol"].tolist()[:K_SLOTS]
            for d, g in df.groupby("date_str", sort=False)
        }

    def _universe_for_date(self, date_str: str) -> list[str]:
        ""
        if self._fixed_universe is not None:
            return self._fixed_universe
        syms = self._universe_by_date.get(date_str, [])
        if len(syms) < K_SLOTS:
            syms = syms + ["__PAD__"] * (K_SLOTS - len(syms))
        return syms

    @property
    def symbols(self) -> list[str]:
        ""
        return self._universe_for_date(self._current_date())

    def _read_default_universe(self) -> list[str]:
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT symbol FROM mm_active_picks ORDER BY rank LIMIT ?",
                (K_SLOTS,),
            ).fetchall()
        return [r[0] for r in rows]

    def _reset_state(self) -> None:
        self.state = PortfolioState()
        self.step_idx = 0
        self.episode_start_idx = 0

    def seed_portfolio(
        self,
        cash: float,
        positions: dict[str, tuple[float, float, int]],
    ) -> None:
        ""
        self.state.cash = float(cash)
        self.state.positions = {
            sym: Position(shares=float(sh), avg_cost=float(ac),
                          entry_step=self.step_idx - max(int(dh), 0))
            for sym, (sh, ac, dh) in positions.items()
        }
        pv = self.state.port_value(self._current_prices(self._current_date()))
        self.state.port_value_series = [pv]
        self.state.running_max = pv


    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        max_start = len(self.trading_dates) - self.episode_len - 1
        self.episode_start_idx = int(self._rng.integers(0, max_start + 1))
        self.step_idx = 0
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"invalid action {action}"
        prev_port_value = self.state.port_value_series[-1]

        turnover = self._apply_action(int(action))

        self.step_idx += 1
        current_date = self._current_date()
        prices = self._current_prices(current_date)
        new_port_value = self.state.port_value(prices)
        self.state.port_value_series.append(new_port_value)
        self.state.running_max = max(self.state.running_max, new_port_value)
        r_step = np.log(max(new_port_value, 1e-6) / max(prev_port_value, 1e-6))
        self.state.step_returns.append(float(r_step))

        reward, reward_parts = self._compute_reward(r_step, turnover, new_port_value)

        current_dd = (new_port_value - self.state.running_max) / self.state.running_max
        terminated = current_dd <= DD_TERMINATE
        truncated = self.step_idx >= self.episode_len

        if terminated or truncated:
            terminal_bonus = self._terminal_reward()
            reward += terminal_bonus
            reward_parts["r_term"] = terminal_bonus

        self.state.prev_action = int(action)
        info = self._info() | {"reward_parts": reward_parts, "current_dd": float(current_dd)}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info


    def _apply_action(self, action: int) -> float:
        ""
        turnover = 0.0
        if action == 0:
            return 0.0
        next_date = self._peek_next_date()
        if next_date is None:
            return 0.0
        if 1 <= action <= K_SLOTS:
            slot = action - 1
            sym = self.symbols[slot]
            if sym == "__PAD__":
                return 0.0
            if sym in self.state.positions and self.state.positions[sym].shares > 0:
                return 0.0
            price = self.data.open_price(sym, next_date)
            if price is None or price <= 0:
                return 0.0
            port_value_now = self.state.port_value(self._current_prices(self._current_date()))
            notional = POSITION_SIZE_FRAC * port_value_now
            cost = notional * (1 + FEE_BPS / 1e4)
            if cost > self.state.cash:
                return 0.0
            shares = notional / price
            self.state.positions[sym] = Position(
                shares=shares, avg_cost=price, entry_step=self.step_idx,
            )
            self.state.cash -= cost
            turnover = notional
        elif K_SLOTS + 1 <= action <= 2 * K_SLOTS:
            held_syms = sorted(
                s for s, p in self.state.positions.items() if p.shares > 0
            )
            idx = action - K_SLOTS - 1
            if idx >= len(held_syms):
                return 0.0
            sym = held_syms[idx]
            pos = self.state.positions[sym]
            price = self.data.open_price(sym, next_date)
            if price is None or price <= 0:
                return 0.0
            notional = pos.shares * price
            proceeds = notional * (1 - FEE_BPS / 1e4)
            self.state.cash += proceeds
            del self.state.positions[sym]
            turnover = notional
        return float(turnover)


    def _compute_reward(
        self, r_step: float, turnover: float, port_value: float,
    ) -> tuple[float, dict[str, float]]:
        recent = self.state.step_returns[-30:]
        vol_30d = float(np.std(recent)) if len(recent) >= 2 else 0.0
        r_vol = -LAMBDA_VOL * vol_30d


        current_dd = (port_value - self.state.running_max) / self.state.running_max
        dd_excess = max(0.0, DD_THRESHOLD - current_dd)
        r_dd = -LAMBDA_DD * max(0.0, dd_excess - self.state.prev_dd_excess)
        self.state.prev_dd_excess = dd_excess

        prices = self._current_prices(self._current_date())
        sector_weights: dict[str, float] = {}
        for sym, pos in self.state.positions.items():
            v = pos.shares * prices.get(sym, pos.avg_cost)
            sec = self.data.sector(sym)
            sector_weights[sec] = sector_weights.get(sec, 0.0) + v
        total = max(port_value, 1e-6)
        hhi = sum((w / total) ** 2 for w in sector_weights.values())
        hhi_excess = max(0.0, hhi - HHI_THRESHOLD)
        r_conc = -LAMBDA_CONC * max(0.0, hhi_excess - self.state.prev_hhi_excess)
        self.state.prev_hhi_excess = hhi_excess

        r_cost = -(FEE_BPS * (turnover / max(port_value, 1e-6))) / 1e4

        reward = r_step + r_vol + r_dd + r_conc + r_cost
        return reward, {
            "r_ret": float(r_step), "r_vol": float(r_vol),
            "r_dd": float(r_dd), "r_conc": float(r_conc), "r_cost": float(r_cost),
        }

    def _terminal_reward(self) -> float:
        returns = np.array(self.state.step_returns, dtype=np.float64)
        if len(returns) < 2 or returns.std() < 1e-9:
            return 0.0
        annualized_sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        return float(BETA_TERM * annualized_sharpe)


    def _current_date(self) -> str:
        return self.trading_dates[self.episode_start_idx + self.step_idx]

    def _peek_next_date(self) -> str | None:
        idx = self.episode_start_idx + self.step_idx + 1
        if idx >= len(self.trading_dates):
            return None
        return self.trading_dates[idx]

    def _current_prices(self, date_str: str) -> dict[str, float]:
        universe = [s for s in self._universe_for_date(date_str) if s != "__PAD__"]
        held = list(self.state.positions.keys())
        all_syms = set(universe) | set(held)
        return {
            s: self.data.close_price(s, date_str) or 0.0
            for s in all_syms
        }

    def _observation(self) -> np.ndarray:
        d = self._current_date()
        prices = self._current_prices(d)
        port_value = self.state.port_value(prices)
        recent = self.state.step_returns[-30:]
        sigma_30d = float(np.std(recent)) if len(recent) >= 2 else 0.0
        running_dd = (port_value - self.state.running_max) / self.state.running_max
        portfolio_state = np.array([
            self.state.cash / max(port_value, 1e-6),
            len([p for p in self.state.positions.values() if p.shares > 0]) / K_SLOTS,
            np.log(max(port_value / INIT_CASH, 1e-6)),
            running_dd,
            sigma_30d,
            (self.episode_len - self.step_idx) / self.episode_len,
            self.state.prev_action / ACTION_DIM,
        ], dtype=np.float32)
        reg = self.data.regime_row(d)
        if reg is None:
            regime = np.zeros(OBS_DIM_REGIME, dtype=np.float32)
        else:
            spy, vix, vixreg, fomc = reg
            def _num(x, default=0.0):
                if x is None:
                    return default
                try:
                    v = float(x)
                    return v if np.isfinite(v) else default
                except (TypeError, ValueError):
                    return default
            regime = np.array([
                _num(spy),
                _num(vix) / 30.0 - 1.0 if _num(vix) > 0 else 0.0,
                1.0 if vixreg == "low" else 0.0,
                1.0 if vixreg == "high" else 0.0,
                _num(fomc),
            ], dtype=np.float32)
        slots = np.zeros((K_SLOTS, OBS_DIM_PER_SLOT), dtype=np.float32)
        for i, sym in enumerate(self.symbols):
            if sym == "__PAD__":
                continue
            pos = self.state.positions.get(sym)
            held = 1.0 if (pos and pos.shares > 0) else 0.0
            days_held = (
                (self.step_idx - pos.entry_step) / self.episode_len
                if (pos and pos.shares > 0) else 0.0
            )
            unrealized = 0.0
            if pos and pos.shares > 0:
                cur = prices.get(sym, pos.avg_cost)
                unrealized = (cur - pos.avg_cost) / max(pos.avg_cost, 1e-6)
            ind = self.data.indicators_row(sym, d) or (0.0, 0.0, 0.0, 0.0)
            def _clean(x):
                if x is None:
                    return 0.0
                fv = float(x)
                return fv if np.isfinite(fv) else 0.0
            rsi, macdh, atr, volz = [_clean(x) for x in ind]
            close = prices.get(sym, 0.0) or 1.0
            atr_pct = atr / max(close, 1e-6) if close else 0.0
            dl_raw = self.data.dl_prediction(sym, d)
            dl_pred = float(dl_raw) if dl_raw is not None and np.isfinite(float(dl_raw)) else 0.0
            sec = self.data.sector(sym)
            sec_onehot = [1.0 if sec == s else 0.0 for s in SECTOR_ORDER]
            slot = [
                held,
                float(days_held),
                float(unrealized),
                0.0,
                rsi / 100.0 if rsi else 0.0,
                float(macdh),
                float(atr_pct),
                float(volz),
                *sec_onehot,
                float(dl_pred),
            ]
            if len(slot) != OBS_DIM_PER_SLOT:
                raise ValueError(
                    f"slot feature length {len(slot)} != OBS_DIM_PER_SLOT "
                    f"{OBS_DIM_PER_SLOT}; update the derived layout, not the data"
                )
            slots[i, :] = slot
        obs = np.concatenate([portfolio_state, regime, slots.flatten()])
        return obs.astype(np.float32)

    def _info(self) -> dict:
        return {
            "date": self._current_date(),
            "step_idx": self.step_idx,
            "port_value": float(self.state.port_value_series[-1]),
            "cash": float(self.state.cash),
            "n_positions": len([p for p in self.state.positions.values() if p.shares > 0]),
        }


    def action_masks(self) -> np.ndarray:
        ""
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[0] = True
        next_date = self._peek_next_date()
        if next_date is None:
            return mask
        current_date = self._current_date()
        prices = self._current_prices(current_date)
        port_value_now = self.state.port_value(prices)
        notional = POSITION_SIZE_FRAC * port_value_now
        buy_cost = notional * (1 + FEE_BPS / 1e4)
        symbols_today = self.symbols
        for slot, sym in enumerate(symbols_today):
            if sym == "__PAD__":
                continue
            open_price = self.data.open_price(sym, next_date)
            if open_price is None or open_price <= 0:
                continue
            held = sym in self.state.positions and self.state.positions[sym].shares > 0
            if not held and buy_cost <= self.state.cash:
                mask[1 + slot] = True
        held_syms = sorted(
            s for s, p in self.state.positions.items() if p.shares > 0
        )
        for i, sym in enumerate(held_syms):
            if i >= K_SLOTS:
                break
            open_price = self.data.open_price(sym, next_date)
            if open_price is None or open_price <= 0:
                continue
            mask[K_SLOTS + 1 + i] = True
        return mask
