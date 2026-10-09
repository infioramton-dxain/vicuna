""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import requests

from macmini.config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
)
from macmini.data._db import connect

log = logging.getLogger(__name__)

_ORDER_PAGE_LIMIT = 500
_HTTP_TIMEOUT_S = 30
_DEFAULT_ORDERS_LOOKBACK_DAYS = 14
_DEFAULT_ACTIVITIES_LOOKBACK_DAYS = 7

_LIVE_ORDER_STATUSES = (
    "new", "held", "accepted", "partially_filled", "pending_new",
    "pending_replace", "pending_cancel", "calculated", "accepted_for_bidding",
)


def _headers() -> dict[str, str]:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing from environment; "
            "cannot pull broker mirror."
        )
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    r = requests.get(
        f"{ALPACA_BASE_URL.rstrip('/')}{path}",
        headers=_headers(),
        params=params or {},
        timeout=_HTTP_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()


def _f(v: Any) -> float | None:
    ""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _b(v: Any) -> int | None:
    ""
    if v is None:
        return None
    return 1 if v else 0


_POSITION_COLS = (
    "symbol", "side", "qty", "avg_entry_price", "market_value", "cost_basis",
    "current_price", "unrealized_pl", "unrealized_plpc", "change_today",
    "exchange", "asset_class", "snapshot_at",
)


def _row_position(p: dict, snapshot_at: str) -> tuple:
    return (
        p.get("symbol"),
        p.get("side"),
        _f(p.get("qty")),
        _f(p.get("avg_entry_price")),
        _f(p.get("market_value")),
        _f(p.get("cost_basis")),
        _f(p.get("current_price")),
        _f(p.get("unrealized_pl")),
        _f(p.get("unrealized_plpc")),
        _f(p.get("change_today")),
        p.get("exchange"),
        p.get("asset_class"),
        snapshot_at,
    )


def snapshot_positions(conn: sqlite3.Connection, snapshot_at: str) -> int:
    positions = _get("/v2/positions")
    conn.execute("DELETE FROM mm_alpaca_positions")
    if not positions:
        conn.commit()
        return 0
    placeholders = ",".join("?" * len(_POSITION_COLS))
    cols = ",".join(_POSITION_COLS)
    conn.executemany(
        f"INSERT INTO mm_alpaca_positions ({cols}) VALUES ({placeholders})",
        [_row_position(p, snapshot_at) for p in positions],
    )
    conn.commit()
    return len(positions)


_ORDER_COLS = (
    "id", "client_order_id", "symbol", "side", "order_type", "order_class",
    "qty", "filled_qty", "filled_avg_price", "limit_price", "stop_price",
    "time_in_force", "status", "parent_id", "replaced_by", "replaces",
    "extended_hours", "submitted_at", "filled_at", "canceled_at",
    "expired_at", "replaced_at", "updated_at", "snapshot_at",
)


def _iter_orders_flat(orders: Iterable[dict]) -> Iterable[dict]:
    ""
    for o in orders:
        yield o
        for leg in (o.get("legs") or []):
            leg = dict(leg)
            leg.setdefault("parent_id", o.get("id"))
            yield leg


def _row_order(o: dict, snapshot_at: str) -> tuple:
    return (
        o.get("id"),
        o.get("client_order_id"),
        o.get("symbol"),
        o.get("side"),
        o.get("order_type"),
        o.get("order_class"),
        _f(o.get("qty")),
        _f(o.get("filled_qty")),
        _f(o.get("filled_avg_price")),
        _f(o.get("limit_price")),
        _f(o.get("stop_price")),
        o.get("time_in_force"),
        o.get("status"),
        o.get("parent_id"),
        o.get("replaced_by"),
        o.get("replaces"),
        _b(o.get("extended_hours")),
        o.get("submitted_at"),
        o.get("filled_at"),
        o.get("canceled_at"),
        o.get("expired_at"),
        o.get("replaced_at"),
        o.get("updated_at"),
        snapshot_at,
    )


def snapshot_orders(
    conn: sqlite3.Connection,
    snapshot_at: str,
    after_iso: str,
) -> int:
    ""
    seen: set[str] = set()
    placeholders = ",".join("?" * len(_ORDER_COLS))
    cols = ",".join(_ORDER_COLS)
    upsert_sql = (
        f"INSERT OR REPLACE INTO mm_alpaca_orders ({cols}) VALUES ({placeholders})"
    )

    until: str | None = None
    while True:
        params: dict[str, Any] = {
            "status": "all",
            "limit": _ORDER_PAGE_LIMIT,
            "after": after_iso,
            "direction": "desc",
            "nested": "true",
        }
        if until is not None:
            params["until"] = until
        page = _get("/v2/orders", params=params)
        if not page:
            break
        flat = list(_iter_orders_flat(page))
        conn.executemany(upsert_sql, [_row_order(o, snapshot_at) for o in flat])
        conn.commit()
        seen.update(o.get("id") for o in flat)
        if len(page) < _ORDER_PAGE_LIMIT:
            break
        until = page[-1].get("submitted_at")
        if until is None or until <= after_iso:
            break

    open_page = _get("/v2/orders", params={
        "status": "open", "limit": _ORDER_PAGE_LIMIT, "nested": "true",
    }) or []
    flat = list(_iter_orders_flat(open_page))
    if flat:
        conn.executemany(upsert_sql, [_row_order(o, snapshot_at) for o in flat])
        conn.commit()
        seen.update(o.get("id") for o in flat)

    ph = ",".join("?" * len(_LIVE_ORDER_STATUSES))
    stragglers = [r[0] for r in conn.execute(
        f"SELECT id FROM mm_alpaca_orders WHERE status IN ({ph}) "
        f"AND snapshot_at != ?", (*_LIVE_ORDER_STATUSES, snapshot_at))]
    for oid in stragglers:
        try:
            o = _get(f"/v2/orders/{oid}")
        except Exception as exc:
            log.warning("mirror: re-fetch of straggler order %s failed: %s", oid, exc)
            continue
        if not o:
            continue
        if not o.get("parent_id"):
            prev = conn.execute(
                "SELECT parent_id FROM mm_alpaca_orders WHERE id = ?", (oid,)).fetchone()
            if prev and prev[0]:
                o = dict(o, parent_id=prev[0])
        conn.execute(upsert_sql, _row_order(o, snapshot_at))
        conn.commit()
        seen.add(oid)
    if stragglers:
        log.info("mirror: refreshed %d straggler order(s) outside the window", len(stragglers))
    return len(seen)


_ACTIVITY_COLS = (
    "id", "activity_type", "transaction_time", "symbol", "side", "qty",
    "price", "cum_qty", "leaves_qty", "order_id", "order_status",
    "net_amount", "per_share_amount", "description", "snapshot_at",
)


def _row_activity(a: dict, snapshot_at: str) -> tuple:
    return (
        a.get("id"),
        a.get("activity_type"),
        a.get("transaction_time") or a.get("date"),
        a.get("symbol"),
        a.get("side"),
        _f(a.get("qty")),
        _f(a.get("price")),
        _f(a.get("cum_qty")),
        _f(a.get("leaves_qty")),
        a.get("order_id"),
        a.get("order_status"),
        _f(a.get("net_amount")),
        _f(a.get("per_share_amount")),
        a.get("description"),
        snapshot_at,
    )


def _iter_dates(start: date, end_inclusive: date) -> Iterable[date]:
    d = start
    while d <= end_inclusive:
        yield d
        d += timedelta(days=1)


def snapshot_activities(
    conn: sqlite3.Connection,
    snapshot_at: str,
    since_date: date,
    until_date: date | None = None,
) -> int:
    ""
    end = until_date or date.today()
    placeholders = ",".join("?" * len(_ACTIVITY_COLS))
    cols = ",".join(_ACTIVITY_COLS)
    ignore_sql = (
        f"INSERT OR IGNORE INTO mm_alpaca_activities ({cols}) "
        f"VALUES ({placeholders})"
    )
    written = 0
    for d in _iter_dates(since_date, end):
        acts = _get("/v2/account/activities", params={"date": d.isoformat()})
        if not acts:
            continue
        conn.executemany(ignore_sql, [_row_activity(a, snapshot_at) for a in acts])
        conn.commit()
        written += len(acts)
    return written


def snapshot(
    db_path: str | None = None,
    orders_since: str | None = None,
    activities_since: str | None = None,
) -> dict:
    ""
    snapshot_at = datetime.now(timezone.utc).isoformat()
    today = date.today()
    orders_after_iso = (
        orders_since
        or (today - timedelta(days=_DEFAULT_ORDERS_LOOKBACK_DAYS)).isoformat() + "T00:00:00Z"
    )
    activities_since_d = (
        date.fromisoformat(activities_since)
        if activities_since
        else today - timedelta(days=_DEFAULT_ACTIVITIES_LOOKBACK_DAYS)
    )

    with connect(db_path) as conn:
        n_pos = snapshot_positions(conn, snapshot_at)
        n_ord = snapshot_orders(conn, snapshot_at, orders_after_iso)
        n_act = snapshot_activities(conn, snapshot_at, activities_since_d)

    log.info(
        "broker mirror snapshot: positions=%d orders=%d activities=%d at %s",
        n_pos, n_ord, n_act, snapshot_at,
    )
    return {
        "positions":  n_pos,
        "orders":     n_ord,
        "activities": n_act,
        "snapshot_at": snapshot_at,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(
        description="Snapshot Alpaca paper broker state into mm_alpaca_* tables."
    )
    p.add_argument("--db", default=None)
    p.add_argument("--orders-since", default=None,
                   help="ISO timestamp; default: 14 days ago")
    p.add_argument("--activities-since", default=None,
                   help="YYYY-MM-DD; default: 7 days ago")
    args = p.parse_args(argv)
    out = snapshot(
        db_path=args.db,
        orders_since=args.orders_since,
        activities_since=args.activities_since,
    )
    print("alpaca broker mirror done:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
