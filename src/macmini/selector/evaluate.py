""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macmini.selector.panel import (MARKET_FEATURES, MAX_HOLD, OUTCOME,
                                    PANEL_PATH, PRICE_FEATURES, REPO_ROOT, STEP)

log = logging.getLogger(__name__)

TEST_YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
EMBARGO_SAMPLES = MAX_HOLD // STEP + 2
TOP_K = 10
COST = 0.0015
SEEDS = (11, 23, 47)
PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.03,
              subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
              reg_lambda=5.0, tree_method="hist")
MIN_TRAIN_PRICE = 5.0
ACCEPT = {"t_excess": 2.5, "years_positive": 5, "t_ic": 3.0}

REPORT_DIR = REPO_ROOT / "exports"


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in PRICE_FEATURES + MARKET_FEATURES if c in panel.columns] + [
        c for c in panel.columns if c.startswith("fund_")]


def prepare(panel: pd.DataFrame, feats: list[str], objective: str = "rank") -> pd.DataFrame:
    ""
    df = panel[panel["close"] >= MIN_TRAIN_PRICE].copy()
    per_stock = [f for f in feats if f not in MARKET_FEATURES]
    df[per_stock] = df.groupby("date")[per_stock].rank(pct=True)
    if objective == "hit":
        df["y"] = (df[OUTCOME] > 0).astype(float).where(df[OUTCOME].notna())
    else:
        df["y"] = df.groupby("date")[OUTCOME].rank(pct=True)
    return df


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                objective: str = "rank") -> np.ndarray:
    import xgboost as xgb

    from macmini.dl.device import require_cuda
    device = require_cuda("selector training")
    tr = train[train["y"].notna()]
    X, y = tr[feats].to_numpy(np.float32), tr["y"].to_numpy(np.float32)
    Xt = test[feats].to_numpy(np.float32)
    pred = np.zeros(len(test), dtype=np.float64)
    for seed in SEEDS:
        if objective == "hit":
            model = xgb.XGBClassifier(**PARAMS, device=device, random_state=seed)
            model.fit(X, y)
            pred += model.predict_proba(Xt)[:, 1]
        else:
            model = xgb.XGBRegressor(**PARAMS, device=device, random_state=seed)
            model.fit(X, y)
            pred += model.predict(Xt)
    return pred / len(SEEDS)


def walk_forward(df: pd.DataFrame, feats: list[str], objective: str = "rank",
                 train_df: pd.DataFrame | None = None) -> pd.DataFrame:
    ""
    dates = np.sort(df["date"].unique())
    pool = df if train_df is None else train_df
    pool_dates = np.sort(pool["date"].unique())
    embargo = pd.Timedelta(days=int((MAX_HOLD + 2) * 7 / 5) + 5)
    out = []
    for year in TEST_YEARS:
        test_dates = dates[pd.DatetimeIndex(dates).year == year]
        if not len(test_dates):
            continue
        train_dates = pool_dates[pool_dates < test_dates[0] - embargo]
        train = pool[pool["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)].copy()
        test["score"] = fit_predict(train, test, feats, objective)
        log.info("%d: trained on %d rows (to %s), scored %d", year, len(train),
                 pd.Timestamp(train_dates[-1]).date(), len(test))
        out.append(test)
    return pd.concat(out, ignore_index=True)


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def score_card(scored: pd.DataFrame, column: str = "score", k: int = TOP_K,
               ascending: bool = False) -> dict:
    ""
    e = scored[scored["eligible"] & scored[OUTCOME].notna()].copy()
    e["bw"] = ((e["br"] - COST) / e["stop_frac"]).clip(-2.0, 3.0)
    e["down"] = (e["how"] == "stop") & (e["f40"] < 0)
    e["win"] = e["br"] > COST
    rows = []
    for d, g in e.groupby("date"):
        if len(g) < 3 * k:
            continue
        top = g.sort_values(column, ascending=ascending).head(k)
        rows.append({
            "date": d, "top": top["bw"].mean(), "rand": g["bw"].mean(),
            "top_win": top["win"].mean(), "rand_win": g["win"].mean(),
            "top_down": top["down"].mean(), "rand_down": g["down"].mean(),
            "top_stop": (top["how"] == "stop").mean(),
            "ic": g[column].corr(g[OUTCOME], method="spearman") * (-1 if ascending else 1),
        })
    r = pd.DataFrame(rows)
    r["excess"] = r["top"] - r["rand"]
    r["year"] = pd.DatetimeIndex(r["date"]).year
    r["quarter"] = pd.DatetimeIndex(r["date"]).to_period("Q")
    by_year = r.groupby("year").agg(
        dates=("excess", "size"), top=("top", "mean"), random=("rand", "mean"),
        excess=("excess", "mean"), top_win=("top_win", "mean"),
        random_win=("rand_win", "mean"), top_down=("top_down", "mean"),
        random_down=("rand_down", "mean"))
    return {
        "dates": int(len(r)), "k": k,
        "top_mean_net": float(r["top"].mean()),
        "random_mean_net": float(r["rand"].mean()),
        "excess": float(r["excess"].mean()),
        "t_excess_quarterly": _tstat(r.groupby("quarter")["excess"].mean()),
        "t_excess_by_date": _tstat(r["excess"]),
        "years_positive": int((by_year["excess"] > 0).sum()),
        "years_tested": int(len(by_year)),
        "top_win": float(r["top_win"].mean()), "random_win": float(r["rand_win"].mean()),
        "top_stays_down": float(r["top_down"].mean()),
        "random_stays_down": float(r["rand_down"].mean()),
        "top_stop_rate": float(r["top_stop"].mean()),
        "rank_ic": float(r["ic"].mean()), "t_ic": _tstat(r["ic"]),
        "by_year": {int(y): {k2: float(v) for k2, v in row.items()}
                    for y, row in by_year.iterrows()},
    }


def verdict(card: dict) -> dict:
    checks = {
        "1_excess_positive_t>=2.5": card["excess"] > 0
        and card["t_excess_quarterly"] >= ACCEPT["t_excess"],
        "2_positive_in_5_of_7_years": card["years_positive"] >= ACCEPT["years_positive"],
        "3_stays_down_not_worse": card["top_stays_down"] <= card["random_stays_down"],
        "4_rank_ic_t>=3": card["rank_ic"] > 0 and card["t_ic"] >= ACCEPT["t_ic"],
    }
    return {"checks": checks, "accepted": bool(all(checks.values()))}


def run(panel_path: Path | str = PANEL_PATH, objective: str = "rank",
        train_panel_path: Path | str | None = None) -> dict:
    panel = pd.read_parquet(panel_path)
    feats = feature_columns(panel)
    df = prepare(panel, feats, objective)
    train_df = prepare(pd.read_parquet(train_panel_path), feats, objective) \
        if train_panel_path else None
    scored = walk_forward(df, feats, objective, train_df)
    card = score_card(scored)
    references = {
        "low_volatility": score_card(scored, "vol_60", ascending=True),
        "momentum_12_1": score_card(scored, "mom_12_1"),
        "one_month_reversal": score_card(scored, "ret_21", ascending=True),
        "near_52w_high": score_card(scored, "off_high_252"),
    }
    post = scored[scored["date"] >= "2026-07-29"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel": str(panel_path), "train_panel": str(train_panel_path or panel_path),
        "objective": objective, "features": feats, "params": PARAMS,
        "seeds": list(SEEDS), "test_years": list(TEST_YEARS),
        "embargo_samples": EMBARGO_SAMPLES, "acceptance": ACCEPT,
        "selector": card, "verdict": verdict(card),
        "top_5": score_card(scored, k=5), "top_20": score_card(scored, k=20),
        "references": {k: {m: v[m] for m in (
            "excess", "t_excess_quarterly", "years_positive", "top_win",
            "top_stays_down", "rank_ic", "t_ic")} for k, v in references.items()},
        "after_universe_freeze_dates": int(post["date"].nunique()),
    }
    return {"report": report, "scored": scored}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="Walk-forward selector test.")
    p.add_argument("--panel", default=str(PANEL_PATH))
    p.add_argument("--tag", default=datetime.now().strftime("%Y%m%d"))
    p.add_argument("--objective", choices=("rank", "hit"), default="rank")
    p.add_argument("--train-panel", default=None,
                   help="denser panel for fitting only (e.g. built with --step 1)")
    args = p.parse_args(argv)
    res = run(args.panel, args.objective, args.train_panel)
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"selector_eval_{args.tag}.json"
    out.write_text(json.dumps(res["report"], indent=1, default=str))
    res["scored"].to_parquet(Path(args.panel).with_name(f"scored_{args.tag}.parquet"),
                             index=False)
    c, v = res["report"]["selector"], res["report"]["verdict"]
    print(json.dumps({k: c[k] for k in c if k != "by_year"}, indent=1))
    print(pd.DataFrame(c["by_year"]).T.round(4).to_string())
    print(json.dumps(res["report"]["references"], indent=1))
    print(json.dumps(v, indent=1), "→", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
