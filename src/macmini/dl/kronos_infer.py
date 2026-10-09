""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
KRONOS_DIR = REPO_ROOT / "third_party" / "kronos"

HF_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
HF_MODEL = "NeoQuasar/Kronos-base"

DEFAULT_MODEL_ID = "kronos_base_zs_t10p90s3"
MAX_CONTEXT = 512
PRED_LEN = 30
N_SAMPLES = 3
MIN_CONTEXT = 64
TEMPERATURE = 1.0
TOP_P = 0.9
SEED = 20260724

SANE_5D_ABS = 0.5
SANE_30D_ABS = 1.0
FLAG_OK = "ok"
FLAG_DEGENERATE = "degenerate"


def classify_forecast(mean_5d: float, mean_30d: float, vol_5d: float,
                      n_samples: int) -> str:
    ""
    if abs(mean_5d) > SANE_5D_ABS or abs(mean_30d) > SANE_30D_ABS:
        return FLAG_DEGENERATE
    if n_samples > 1 and vol_5d == 0.0:
        return FLAG_DEGENERATE
    return FLAG_OK


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_symbol_bars(conn: sqlite3.Connection, symbol: str,
                     max_context: int = MAX_CONTEXT) -> pd.DataFrame:
    ""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM mm_bars "
        "WHERE symbol = ? ORDER BY date DESC LIMIT ?",
        conn, params=(symbol, max_context))
    return df.iloc[::-1].reset_index(drop=True)


class KronosForecaster:
    ""

    _instance: "KronosForecaster | None" = None

    def __init__(self, device: str | None = None):
        import sys
        if str(KRONOS_DIR) not in sys.path:
            sys.path.insert(0, str(KRONOS_DIR))
        from model import Kronos, KronosPredictor, KronosTokenizer

        from macmini.dl.device import require_cuda

        if device is None:
            device = require_cuda("Kronos inference")
        tokenizer = KronosTokenizer.from_pretrained(HF_TOKENIZER)
        model = Kronos.from_pretrained(HF_MODEL)
        self.device = device
        self.predictor = KronosPredictor(model, tokenizer, device=device,
                                         max_context=MAX_CONTEXT)

    @classmethod
    def get(cls, device: str | None = None) -> "KronosForecaster":
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def forecast_symbols(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        pred_len: int = PRED_LEN,
        n_samples: int = N_SAMPLES,
        seed: int = SEED,
    ) -> dict[str, dict]:
        ""
        import torch
        torch.manual_seed(seed)

        by_len: dict[int, list[str]] = {}
        y_dates_by_symbol: dict[str, pd.Series] = {}
        for sym, bars in bars_by_symbol.items():
            if len(bars) < MIN_CONTEXT:
                continue
            by_len.setdefault(len(bars), []).append(sym)
        if not by_len:
            return {}

        symbols, preds = [], []
        for length, group in by_len.items():
            g_syms, df_list, x_ts_list, y_ts_list = [], [], [], []
            for sym in group:
                bars = bars_by_symbol[sym]
                x_df = bars[["open", "high", "low", "close", "volume"]]
                x_ts = pd.to_datetime(bars["date"])
                y_ts = pd.Series(pd.bdate_range(x_ts.iloc[-1], periods=pred_len + 1)[1:])
                y_dates_by_symbol[sym] = y_ts
                for _ in range(n_samples):
                    g_syms.append(sym)
                    df_list.append(x_df)
                    x_ts_list.append(x_ts)
                    y_ts_list.append(y_ts)
            symbols.extend(g_syms)
            preds.extend(self.predictor.predict_batch(
                df_list, x_ts_list, y_ts_list, pred_len=pred_len,
                T=TEMPERATURE, top_p=TOP_P, sample_count=1, verbose=False))

        out: dict[str, dict] = {}
        for sym in y_dates_by_symbol:
            sample_dfs = [p for s, p in zip(symbols, preds) if s == sym]
            closes = np.stack([p["close"].to_numpy() for p in sample_dfs])
            last_close = float(bars_by_symbol[sym]["close"].iloc[-1])
            mean_path = (
                sum(p[["open", "high", "low", "close", "volume"]] for p in sample_dfs)
                / len(sample_dfs)
            )
            k5 = min(5, pred_len)
            out[sym] = {
                "as_of_date": str(bars_by_symbol[sym]["date"].iloc[-1]),
                "y_dates": y_dates_by_symbol[sym].dt.strftime("%Y-%m-%d"),
                "mean_path": mean_path,
                "close_lo": closes.min(axis=0),
                "close_hi": closes.max(axis=0),
                "sample_5d_returns": closes[:, k5 - 1] / last_close - 1.0,
                "sample_30d_returns": closes[:, -1] / last_close - 1.0,
            }
        return out


def persist_forecast(db_path: Path | str, symbol: str, fc: dict,
                     model_id: str = DEFAULT_MODEL_ID,
                     n_samples: int = N_SAMPLES) -> str:
    ""
    now = _now_iso()
    r5 = fc["sample_5d_returns"]
    r30 = fc["sample_30d_returns"]
    mean_5d, mean_30d, vol_5d = float(np.mean(r5)), float(np.mean(r30)), float(np.std(r5))
    flag = classify_forecast(mean_5d, mean_30d, vol_5d, n_samples)
    mp = fc["mean_path"]
    path_rows = [
        (model_id, symbol, fc["as_of_date"], step, d,
         float(row.open), float(row.high), float(row.low), float(row.close),
         float(row.volume), float(lo), float(hi), now)
        for step, (d, row, lo, hi) in enumerate(
            zip(fc["y_dates"], mp.itertuples(index=False),
                fc["close_lo"], fc["close_hi"]), start=1)
    ]
    with sqlite3.connect(db_path, timeout=30) as c:
        c.execute("PRAGMA busy_timeout = 30000")
        c.execute("PRAGMA journal_mode = WAL")
        c.execute(
            "INSERT OR REPLACE INTO mm_kronos_signals "
            "(model_id, symbol, date, predicted_5d_return, predicted_30d_return, "
            " predicted_5d_vol, sample_count, quality_flag, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (model_id, symbol, fc["as_of_date"], mean_5d, mean_30d, vol_5d,
             n_samples, flag, now))
        c.execute(
            "DELETE FROM mm_kronos_paths WHERE model_id=? AND symbol=? AND as_of_date=?",
            (model_id, symbol, fc["as_of_date"]))
        c.executemany(
            "INSERT INTO mm_kronos_paths "
            "(model_id, symbol, as_of_date, step, date, open, high, low, close, "
            " volume, close_lo, close_hi, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            path_rows)
        c.commit()
    return flag


def get_cached_forecast(db_path: Path | str, symbol: str,
                        model_id: str = DEFAULT_MODEL_ID
                        ) -> tuple[dict | None, pd.DataFrame | None]:
    ""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as c:
        sig = c.execute(
            "SELECT model_id, symbol, date, predicted_5d_return, predicted_30d_return, "
            "predicted_5d_vol, sample_count, created_at FROM mm_kronos_signals "
            "WHERE model_id=? AND symbol=? ORDER BY date DESC LIMIT 1",
            (model_id, symbol)).fetchone()
        if sig is None:
            return None, None
        cols = ["model_id", "symbol", "date", "predicted_5d_return",
                "predicted_30d_return", "predicted_5d_vol", "sample_count",
                "created_at"]
        sig_row = dict(zip(cols, sig))
        paths = pd.read_sql_query(
            "SELECT step, date, open, high, low, close, volume, close_lo, close_hi "
            "FROM mm_kronos_paths WHERE model_id=? AND symbol=? AND as_of_date=? "
            "ORDER BY step", c, params=(model_id, symbol, sig_row["date"]))
    return sig_row, paths


def forecast_and_store(
    db_path: Path | str,
    symbols: Sequence[str],
    model_id: str = DEFAULT_MODEL_ID,
    pred_len: int = PRED_LEN,
    n_samples: int = N_SAMPLES,
    chunk_symbols: int = 20,
    seed: int = SEED,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    ""
    t0 = time.time()
    fx = KronosForecaster.get()
    written, skipped, degenerate = 0, 0, 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as ro:
        bars_all = {s: load_symbol_bars(ro, s) for s in symbols}
    for i in range(0, len(symbols), chunk_symbols):
        chunk = {s: bars_all[s] for s in symbols[i:i + chunk_symbols]}
        results = fx.forecast_symbols(chunk, pred_len=pred_len,
                                      n_samples=n_samples, seed=seed)
        for sym in chunk:
            if sym in results:
                flag = persist_forecast(db_path, sym, results[sym],
                                        model_id=model_id, n_samples=n_samples)
                written += 1
                degenerate += int(flag == FLAG_DEGENERATE)
            else:
                skipped += 1
        if progress_cb:
            progress_cb(min(i + chunk_symbols, len(symbols)), len(symbols))
    return {
        "model_id": model_id, "symbols_written": written,
        "symbols_skipped": skipped, "symbols_degenerate": degenerate,
        "pred_len": pred_len,
        "n_samples": n_samples, "temperature": TEMPERATURE, "top_p": TOP_P,
        "seed": seed, "elapsed_s": round(time.time() - t0, 1),
    }
