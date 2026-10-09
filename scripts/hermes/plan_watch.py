#!/usr/bin/env python3
""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HOME = os.path.expanduser("~")
DEFAULT_DB = os.environ.get("AGENT_SHIPPED_DB", f"{HOME}/agent/db/macmini.db")
DEFAULT_ENV = os.environ.get("PLAN_WATCH_ENV", f"{HOME}/agent/.env")
DEFAULT_PLANS = os.environ.get("PLAN_WATCH_PLANS", f"{HOME}/agent/coord/plan_levels.json")
DEFAULT_STATE = os.environ.get("PLAN_WATCH_STATE", f"{HOME}/agent/state/plan_watch_state.json")

ATR_N = 14
DEFAULT_OVERSHOOT_ATR = 1.0
MAX_BAR_AGE_DAYS = 5
BARS_LOOKBACK = 260
PIVOT_K = 3
STOP_ATR_MULT = 2.0
STOP_MIN_PCT = 0.08
DEFAULT_REWARD_MULT = 2.0
ORDER_RUNAWAY_ATR = 3.0
ORDER_RUNAWAY_SESSIONS = 10
HTTP_TIMEOUT_S = 15

TERMINAL = ("invalidated", "expired")
INACTIVE_STATUSES = ("superseded", "filled", "cancelled", "canceled")

D, O, H, L, C = range(5)


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def business_days_between(a: date, b: date) -> int:
    ""
    n = 0
    while a < b:
        a += timedelta(days=1)
        if a.weekday() < 5:
            n += 1
    return n


def load_bars(db_path: str, symbol: str, n: int = BARS_LOOKBACK) -> list[tuple]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT date, open, high, low, close FROM mm_bars "
            "WHERE symbol = ? AND close IS NOT NULL ORDER BY date DESC LIMIT ?",
            (symbol, n)).fetchall()
    finally:
        con.close()
    return [(r[0][:10], r[1], r[2], r[3], r[4]) for r in rows[::-1]]


def atr(bars: list[tuple], n: int = ATR_N) -> float | None:
    if len(bars) < n + 1:
        return None
    trs = []
    for prev, cur in zip(bars[-n - 1:-1], bars[-n:]):
        trs.append(max(cur[H] - cur[L], abs(cur[H] - prev[C]), abs(cur[L] - prev[C])))
    return sum(trs) / n


def sma(bars: list[tuple], n: int) -> float | None:
    if len(bars) < n:
        return None
    return sum(b[C] for b in bars[-n:]) / n


def rolling_low(bars: list[tuple], n: int, exclude_recent: int = 0) -> tuple[float, str] | None:
    window = bars[-n:] if exclude_recent == 0 else bars[-n:-exclude_recent]
    if not window:
        return None
    b = min(window, key=lambda x: x[L])
    return b[L], b[D]


def rolling_high(bars: list[tuple], n: int) -> tuple[float, str] | None:
    window = bars[-n:]
    if not window:
        return None
    b = max(window, key=lambda x: x[H])
    return b[H], b[D]


def pivot_lows(bars: list[tuple], k: int = PIVOT_K) -> list[tuple[str, float]]:
    out = []
    for i in range(k, len(bars) - k):
        lo = bars[i][L]
        if lo <= min(b[L] for b in bars[i - k:i + k + 1]):
            out.append((bars[i][D], lo))
    return out


def find_floor(bars: list[tuple], price: float, atr_now: float, lookback: int = 252) -> dict | None:
    ""
    pivots = sorted((p for p in pivot_lows(bars[-lookback:]) if p[1] < price),
                    key=lambda p: -p[1])
    if not pivots:
        return None
    clusters, cur = [], [pivots[0]]
    for p in pivots[1:]:
        if cur[0][1] - p[1] <= atr_now:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    chosen = next((c for c in clusters if len(c) >= 2), clusters[0])
    lows = [p[1] for p in chosen]
    return {"lo": min(lows), "hi": max(lows), "touches": len(chosen),
            "entry": round(statistics.median(lows), 2),
            "dates": sorted(p[0] for p in chosen)}


def reanalyse(bars: list[tuple], price: float, target_pct: float | None = None) -> dict:
    out: dict = {"as_of": bars[-1][D] if bars else None, "price": price}
    a = atr(bars)
    if a is None:
        out["verdict"] = "not enough bars to derive a level"
        return out
    out["atr"] = a
    for n in (20, 50, 200):
        out[f"sma{n}"] = sma(bars, n)
    out["low_52w"] = rolling_low(bars, 252)
    out["high_52w"] = rolling_high(bars, 252)
    avgs = [out[f"sma{n}"] for n in (20, 50, 200) if out[f"sma{n}"] is not None]
    below = sum(1 for v in avgs if price < v)
    out["trend"] = ("below all of its 20/50/200-day averages (downtrend)" if below == len(avgs)
                    else "above all of its 20/50/200-day averages (uptrend)" if below == 0
                    else f"below {below} of its {len(avgs)} moving averages (mixed)")
    floor = find_floor(bars, price, a)
    out["floor"] = floor
    if floor is None:
        out["verdict"] = ("no support below the price in the last 252 sessions — the stock "
                          "is at new lows, no entry level can be derived; stand aside")
        return out
    entry = floor["entry"]
    stop_dist = max(STOP_ATR_MULT * a, STOP_MIN_PCT * entry)
    stop = round(entry - stop_dist, 2)
    target = round(entry * (1 + target_pct), 2) if target_pct else round(
        entry + DEFAULT_REWARD_MULT * stop_dist, 2)
    out.update(entry=entry, stop=stop, target=target,
               rr=(target - entry) / (entry - stop),
               verdict="weak floor (single touch)" if floor["touches"] < 2 else "ok")
    return out


def reanalysis_text(symbol: str, r: dict) -> str:
    lines = [f"Re-analysis of {symbol} from current data (daily bars through {r.get('as_of')}, "
             f"live ${r['price']:.2f}):"]
    if "atr" not in r:
        return lines[0] + f"\n  {r['verdict']}."
    lines.append(f"  Trend: {r['trend']}.")
    avg = " · ".join(f"{n}d avg ${r[f'sma{n}']:.2f}" for n in (20, 50, 200)
                     if r.get(f"sma{n}") is not None)
    lines.append(f"  {avg} · typical daily range (ATR{ATR_N}) ${r['atr']:.2f}")
    if r.get("low_52w"):
        lines.append(f"  52-week low ${r['low_52w'][0]:.2f} ({r['low_52w'][1]}), "
                     f"high ${r['high_52w'][0]:.2f} ({r['high_52w'][1]})")
    f = r.get("floor")
    if f is None:
        lines.append(f"  Result: {r['verdict']}.")
        return "\n".join(lines)
    lines.append(f"  Support floor: ${f['lo']:.2f}–${f['hi']:.2f}, {f['touches']} pivot low(s) "
                 f"({', '.join(f['dates'])})" + ("" if r["verdict"] == "ok" else f" — {r['verdict']}"))
    lines.append(f"  Entry (resting limit) ${r['entry']:.2f} · stop ${r['stop']:.2f} · "
                 f"target ${r['target']:.2f} · reward:risk {r['rr']:.1f}:1")
    lines.append("  These are derived levels for a decision, not an order.")
    return "\n".join(lines)


def load_plans(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    raw = json.load(open(path))
    if isinstance(raw, dict) and isinstance(raw.get("plans"), list):
        return raw["plans"]
    plans = []
    for name, spec in (raw.items() if isinstance(raw, dict) else []):
        if isinstance(spec, dict) and "symbol" in spec:
            plans.append({"id": name, "symbol": spec["symbol"], "side": "buy",
                          "set_on": spec.get("set_on"),
                          "anchor": {"kind": "fixed", "price": spec.get("level")},
                          "_legacy": True})
    return plans


def validate(plan: dict) -> list[str]:
    errs = []
    for k in ("id", "symbol", "set_on", "valid_until", "price_when_set"):
        if plan.get(k) in (None, ""):
            errs.append(f"missing `{k}`")
    if plan.get("side", "buy") != "buy":
        errs.append("only `side: buy` is supported (long-only)")
    a = plan.get("anchor") or {}
    if a.get("kind") not in ("fixed", "rolling_low", "sma", "drawdown", "support"):
        errs.append(f"unknown anchor kind {a.get('kind')!r}")
    elif a["kind"] == "fixed" and not a.get("price"):
        errs.append("fixed anchor needs `price`")
    for p in plan.get("premise") or []:
        if p.get("kind") not in ("above_sma", "floor_holds", "no_new_low"):
            errs.append(f"unknown premise kind {p.get('kind')!r}")
    for k in ("set_on", "valid_until"):
        try:
            if plan.get(k):
                date.fromisoformat(plan[k])
        except ValueError:
            errs.append(f"`{k}` is not an ISO date")
    return errs


def derive_level(anchor: dict, bars: list[tuple], price: float) -> tuple[float | None, str]:
    kind = anchor["kind"]
    a = atr(bars)
    if kind == "fixed":
        return float(anchor["price"]), f"fixed price ${float(anchor['price']):.2f}"
    if kind == "rolling_low":
        n, k = int(anchor.get("lookback", 63)), float(anchor.get("offset_atr", 0.0))
        lo = rolling_low(bars, n)
        if lo is None or a is None:
            return None, "not enough bars"
        return lo[0] + k * a, f"{n}-day low ${lo[0]:.2f} ({lo[1]}) {k:+.1f}×ATR"
    if kind == "sma":
        n, pct = int(anchor.get("n", 50)), float(anchor.get("pct", 0.0))
        m = sma(bars, n)
        if m is None:
            return None, "not enough bars"
        return m * (1 + pct), f"{n}-day average ${m:.2f} {pct * 100:+.1f}%"
    if kind == "drawdown":
        n, pct = int(anchor.get("lookback", 252)), float(anchor["pct"])
        hi = rolling_high(bars, n)
        if hi is None:
            return None, "not enough bars"
        return hi[0] * (1 - pct), f"{pct * 100:.0f}% below the {n}-day high ${hi[0]:.2f} ({hi[1]})"
    if a is None:
        return None, "not enough bars"
    f = find_floor(bars, price + a, a, int(anchor.get("lookback", 252)))
    if f is None:
        return None, "no support floor below the market"
    return f["entry"], f"support floor ${f['lo']:.2f}–${f['hi']:.2f} ({f['touches']} pivot lows)"


def check_premise(p: dict, bars: list[tuple]) -> tuple[bool, str]:
    last = bars[-1]
    if p["kind"] == "above_sma":
        n = int(p.get("n", 50))
        m = sma(bars, n)
        if m is None:
            return True, ""
        return last[C] >= m, (f"the {last[D]} close ${last[C]:.2f} is below the {n}-day "
                              f"average ${m:.2f}")
    if p["kind"] == "floor_holds":
        return last[C] >= float(p["price"]), (f"the {last[D]} close ${last[C]:.2f} is below "
                                              f"the floor ${float(p['price']):.2f}")
    n, k = int(p.get("lookback", 252)), int(p.get("exclude_recent", 5))
    lo = rolling_low(bars, n, k)
    if lo is None:
        return True, ""
    return last[C] >= lo[0], (f"the {last[D]} close ${last[C]:.2f} is a new {n}-day low "
                              f"(prior low ${lo[0]:.2f}, {lo[1]})")


def _context(bars: list[tuple]) -> str:
    parts = [f"ATR{ATR_N} ${atr(bars):.2f}"]
    for n in (20, 50):
        m = sma(bars, n)
        if m is not None:
            parts.append(f"{n}d avg ${m:.2f}")
    lo = rolling_low(bars, 252)
    if lo:
        parts.append(f"52w low ${lo[0]:.2f} ({lo[1]})")
    return f"Context as of daily bars {bars[-1][D]}: " + " · ".join(parts)


def evaluate(plan: dict, bars: list[tuple], price: float | None, price_ts: str | None,
             today: date, rec: dict | None) -> tuple[dict, str | None]:
    ""
    rec = dict(rec or {"state": "new"})
    pid, sym = plan.get("id", "?"), plan.get("symbol", "?")
    if rec["state"] in TERMINAL:
        return rec, None

    errs = validate(plan)
    if errs:
        if rec["state"] == "rejected":
            return rec, None
        rec["state"] = "rejected"
        what = ("a bare number with no rule, expiry or premise"
                if plan.get("_legacy") else "incomplete")
        lvl = (plan.get("anchor") or {}).get("price")
        return rec, (f"{sym} — plan '{pid}' NOT ARMED: it is {what}"
                     + (f" (${float(lvl):.2f})" if lvl else "") + ".\n"
                     f"Problems: {'; '.join(errs)}.\n"
                     "Rewrite it in the v2 schema (`plan_watch.py new …`). "
                     "This notice is sent once.")

    stale = (not bars or price is None or atr(bars) is None
             or (today - date.fromisoformat(bars[-1][D])).days > MAX_BAR_AGE_DAYS)
    if stale:
        if rec.get("data_stale"):
            return rec, None
        rec["data_stale"] = True
        last = bars[-1][D] if bars else "none"
        return rec, (f"{sym} — plan '{pid}' cannot be evaluated: "
                     f"{'no live price' if price is None and bars else f'daily bars end {last}'}. "
                     "The plan is paused until data returns. This notice is sent once.")
    rec["data_stale"] = False

    a = atr(bars)
    level, how = derive_level(plan["anchor"], bars, price)
    k = float(plan.get("overshoot_atr", DEFAULT_OVERSHOOT_ATR))
    head = (f"Level rule: {how}; plan set {plan['set_on']} when {sym} was "
            f"${float(plan['price_when_set']):.2f}.")
    order = plan.get("linked_order_id")
    order_line = (f"\nLinked resting order {order} still stands at the broker — review it; "
                  "this script does not cancel orders." if order else "")

    def death(state: str, title: str, why: str) -> tuple[dict, str]:
        rec.update(state=state, reason=why, since=today.isoformat())
        ra = reanalysis_text(sym, reanalyse(bars, price, plan.get("target_pct")))
        return rec, (f"{sym} — plan '{pid}' {title}\n{why}\n{head}\n"
                     f"The plan is retired and will not alert again.{order_line}\n\n{ra}")

    for p in plan.get("premise") or []:
        ok, why = check_premise(p, bars)
        if not ok:
            return death("invalidated", "INVALIDATED: its premise broke",
                         f"Live ${price:.2f} ({price_ts}); {why}.")
    if level is None:
        return death("invalidated", "INVALIDATED: its level can no longer be derived", f"{how}.")
    if price < level - k * a:
        return death("invalidated", "INVALIDATED: the market ran through the level",
                     f"Live ${price:.2f} ({price_ts}) is ${level - price:.2f} below the "
                     f"${level:.2f} level — more than {k:.1f}×ATR (${k * a:.2f}). "
                     "A level this far behind the market is a reason to re-analyse, "
                     "not to buy.")
    if today > date.fromisoformat(plan["valid_until"]):
        return death("expired", f"EXPIRED on {plan['valid_until']}",
                     f"Live ${price:.2f} ({price_ts}); level ${level:.2f}.")

    rec["level"] = round(level, 2)
    if price <= level and rec["state"] != "triggered":
        rec.update(state="triggered", since=today.isoformat())
        return rec, (f"{sym} — plan '{pid}' level reached (first touch)\n"
                     f"Live ${price:.2f} ({price_ts}) is at or below the entry level "
                     f"${level:.2f}.\n{head}\n"
                     f"Valid until {plan['valid_until']}; the plan dies below "
                     f"${level - k * a:.2f} (level − {k:.1f}×ATR).\n{_context(bars)}\n"
                     "This alert is sent once. No order is placed by this script."
                     + (f"\nLinked resting order: {order}" if order else ""))
    if rec["state"] == "new":
        rec.update(state="armed", since=today.isoformat())
    return rec, None


def check_resting_orders(orders: list[dict], prices: dict, bars_by_symbol: dict,
                         today: date, state: dict) -> list[str]:
    ""
    msgs = []
    for o in orders:
        if o.get("side") != "buy" or not o.get("limit_price"):
            continue
        sym, limit = o["symbol"], float(o["limit_price"])
        px = (prices.get(sym) or (None, None))[0]
        a = atr(bars_by_symbol.get(sym) or [])
        if px is None or a is None:
            continue
        age = business_days_between(date.fromisoformat(o["submitted_at"][:10]), today)
        key = f"order:{o['id']}"
        if px > limit + ORDER_RUNAWAY_ATR * a and age >= ORDER_RUNAWAY_SESSIONS:
            if state.get(key, {}).get("state") != "runaway":
                state[key] = {"state": "runaway", "since": today.isoformat()}
                msgs.append(
                    f"{sym} — resting buy {o['id']} looks abandoned by the market\n"
                    f"Limit ${limit:.2f}, live ${px:.2f}: ${px - limit:.2f} above the limit "
                    f"(more than {ORDER_RUNAWAY_ATR:.0f}×ATR ${a:.2f}) after {age} sessions. "
                    "Re-derive the entry or cancel. This notice is sent once.")
        else:
            state.pop(key, None)
    return msgs


def run_watch(plans: list[dict], state: dict, bars_by_symbol: dict, prices: dict,
              orders: list[dict], today: date) -> tuple[list[str], dict]:
    state = json.loads(json.dumps(state))
    plan_state = state.setdefault("plans", {})
    msgs = []
    for plan in plans:
        if str(plan.get("status", "active")).lower() in INACTIVE_STATUSES:
            continue
        sym = plan.get("symbol")
        px, ts = prices.get(sym) or (None, None)
        rec, msg = evaluate(plan, bars_by_symbol.get(sym) or [], px, ts, today,
                            plan_state.get(str(plan.get("id"))))
        plan_state[str(plan.get("id"))] = rec
        if msg:
            msgs.append(msg)
    msgs += check_resting_orders(orders, prices, bars_by_symbol, today,
                                 state.setdefault("orders", {}))
    return msgs, state


def load_env(path: str) -> dict:
    out = dict(os.environ)
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def _get(url: str, env: dict):
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": env.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": env.get("ALPACA_SECRET_KEY", "")})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
        return json.loads(r.read())


def fetch_prices(symbols: list[str], env: dict) -> dict:
    if not symbols:
        return {}
    host = env.get("ALPACA_DATA_URL") or "https://data.alpaca.markets"
    q = _get(f"{host.rstrip('/')}/v2/stocks/trades/latest?symbols={','.join(symbols)}", env)
    return {s: (float(v["p"]), v["t"][:19] + "Z") for s, v in q.get("trades", {}).items()}


def fetch_open_orders(env: dict) -> list[dict]:
    host = (env.get("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
    if host.endswith("/v2"):
        host = host[:-3]
    return _get(f"{host}/v2/orders?status=open&limit=200", env)


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _gather(args, plans: list[dict], with_orders: bool):
    env = load_env(args.env)
    orders = fetch_open_orders(env) if with_orders else []
    symbols = sorted({p["symbol"] for p in plans if p.get("symbol")}
                     | {o["symbol"] for o in orders if o.get("side") == "buy"})
    prices = fetch_prices(symbols, env)
    bars = {s: load_bars(args.db, s) for s in symbols}
    return bars, prices, orders


def cmd_watch(args) -> int:
    state = load_state(args.state)
    try:
        plans = load_plans(args.plans)
        bars, prices, orders = _gather(args, plans, not args.no_orders)
        msgs, new_state = run_watch(plans, state, bars, prices, orders,
                                    datetime.now(ET).date())
        new_state.pop("_error", None)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        if state.get("_error") != err:
            print(f"plan watch ERROR (reported once until it changes): {err}")
            if not args.dry_run:
                state["_error"] = err
                save_state(args.state, state)
        return 1
    if msgs:
        print("\n\n".join(msgs))
    if not args.dry_run:
        save_state(args.state, new_state)
    return 0


def cmd_status(args) -> int:
    plans = load_plans(args.plans)
    state = load_state(args.state).get("plans", {})
    bars, prices, _ = _gather(args, plans, False)
    for p in plans:
        sym, pid = p.get("symbol"), str(p.get("id"))
        px = (prices.get(sym) or (None, None))[0]
        errs = validate(p)
        lvl = None if errs or not bars.get(sym) or px is None else derive_level(
            p["anchor"], bars[sym], px)[0]
        print(f"{pid} | {sym} | file status {p.get('status', 'active')} | "
              f"watch state {state.get(pid, {}).get('state', 'new')} | "
              f"level {'n/a' if lvl is None else f'${lvl:.2f}'} | "
              f"live {'n/a' if px is None else f'${px:.2f}'} | "
              f"valid until {p.get('valid_until')}"
              + (f" | INVALID: {'; '.join(errs)}" if errs else ""))
    return 0


def cmd_reanalyse(args) -> int:
    bars = load_bars(args.db, args.symbol)
    if not bars:
        print(f"no bars for {args.symbol} in {args.db}")
        return 1
    price = args.price
    if price is None:
        price = (fetch_prices([args.symbol], load_env(args.env)).get(args.symbol)
                 or (bars[-1][C], None))[0]
    print(reanalysis_text(args.symbol, reanalyse(bars, price, args.target_pct)))
    return 0


def _parse_kv(spec: str, positional: dict) -> dict:
    ""
    kind, _, rest = spec.partition(":")
    out: dict = {"kind": kind}
    for part in filter(None, rest.split(",")):
        k, eq, v = part.partition("=")
        if not eq:
            k, v = positional[kind], k
        out[k] = float(v) if "." in v or k in ("price", "pct", "offset_atr") else int(v)
    return out


def cmd_new(args) -> int:
    env = load_env(args.env)
    px = args.price or fetch_prices([args.symbol], env)[args.symbol][0]
    today = datetime.now(ET).date()
    plan = {
        "id": args.id or f"{args.symbol.lower()}-{today:%Y%m%d}",
        "symbol": args.symbol, "side": "buy", "status": "active",
        "set_on": today.isoformat(), "price_when_set": round(px, 2),
        "valid_until": add_business_days(today, args.valid_days).isoformat(),
        "anchor": _parse_kv(args.anchor, {"fixed": "price", "rolling_low": "lookback",
                                          "sma": "n", "drawdown": "pct",
                                          "support": "lookback"}),
        "premise": [_parse_kv(p, {"above_sma": "n", "floor_holds": "price",
                                  "no_new_low": "lookback"}) for p in args.premise],
        "overshoot_atr": args.overshoot_atr, "source": args.source,
    }
    if args.target_pct is not None:
        plan["target_pct"] = args.target_pct
    if args.order_id:
        plan["linked_order_id"] = args.order_id
    if args.note:
        plan["note"] = args.note
    errs = validate(plan)
    if errs:
        print(f"invalid plan: {'; '.join(errs)}", file=sys.stderr)
        return 2
    if args.write:
        existing = load_plans(args.plans)
        if any(str(p.get("id")) == plan["id"] for p in existing):
            print(f"plan id {plan['id']!r} already exists — ids are never reused",
                  file=sys.stderr)
            return 2
        existing = [p for p in existing if not p.get("_legacy")]
        os.makedirs(os.path.dirname(args.plans) or ".", exist_ok=True)
        with open(args.plans, "w") as fh:
            json.dump({"version": 2, "plans": existing + [plan]}, fh, indent=1)
    print(json.dumps(plan, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--plans", default=DEFAULT_PLANS)
    ap.add_argument("--state", default=DEFAULT_STATE)
    sub = ap.add_subparsers(dest="cmd")
    w = sub.add_parser("watch")
    w.add_argument("--dry-run", action="store_true")
    w.add_argument("--no-orders", action="store_true")
    sub.add_parser("status")
    r = sub.add_parser("reanalyse")
    r.add_argument("symbol")
    r.add_argument("--price", type=float)
    r.add_argument("--target-pct", type=float)
    n = sub.add_parser("new")
    n.add_argument("--symbol", required=True)
    n.add_argument("--anchor", required=True)
    n.add_argument("--valid-days", type=int, required=True)
    n.add_argument("--premise", action="append", default=[])
    n.add_argument("--overshoot-atr", type=float, default=DEFAULT_OVERSHOOT_ATR)
    n.add_argument("--target-pct", type=float)
    n.add_argument("--price", type=float)
    n.add_argument("--order-id")
    n.add_argument("--id")
    n.add_argument("--source", default="operator")
    n.add_argument("--note")
    n.add_argument("--write", action="store_true", help="append to the plan file")
    argv = sys.argv[1:] if argv is None else argv
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args(list(argv) + ["watch"])
    return {"watch": cmd_watch, "status": cmd_status,
            "reanalyse": cmd_reanalyse, "new": cmd_new}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
