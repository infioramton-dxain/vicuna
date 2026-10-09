""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent.config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    mid: float


@dataclass
class AccountSnapshot:
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float


@dataclass
class Position:
    symbol: str
    qty: int
    market_value: float
    unrealized_pl: float
    avg_entry_price: float


@dataclass
class BracketOrder:
    symbol: str
    qty: int
    side: str
    entry_price: float
    stop_price: float
    target_price: float


@dataclass
class VariantOrder:
    ""
    symbol: str
    qty: int
    side: str
    entry_type: str
    exit_type: str
    entry_price: float
    stop_trigger: float | None
    stop_price: float
    target_price: float | None
    trail_pct: float | None


class AlpacaClient:
    ""

    def __init__(self):
        if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY required")
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        self._trading = TradingClient(
            api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY,
            paper="paper" in (ALPACA_BASE_URL or ""),
        )
        self._data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self._quote_request = StockLatestQuoteRequest

    def get_account(self) -> AccountSnapshot:
        a = self._trading.get_account()
        return AccountSnapshot(
            cash=float(a.cash),
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            portfolio_value=float(a.portfolio_value),
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=p.symbol,
                qty=int(p.qty),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                avg_entry_price=float(p.avg_entry_price),
            )
            for p in self._trading.get_all_positions()
        ]

    def get_quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        symbols = list(symbols)
        if not symbols:
            return {}
        req = self._quote_request(symbol_or_symbols=symbols)
        resp = self._data.get_stock_latest_quote(req)
        out: dict[str, Quote] = {}
        for sym, q in resp.items():
            bid = float(q.bid_price or 0)
            ask = float(q.ask_price or 0)
            mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
            out[sym] = Quote(symbol=sym, bid=bid, ask=ask, mid=mid)
        return out

    def submit_bracket(self, order: BracketOrder) -> str:
        ""
        from alpaca.trading.requests import (
            LimitOrderRequest, StopLossRequest, TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        req = LimitOrderRequest(
            symbol=order.symbol,
            qty=order.qty,
            side=OrderSide.BUY,
            type="limit",
            time_in_force=TimeInForce.GTC,
            limit_price=order.entry_price,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=order.stop_price),
            take_profit=TakeProfitRequest(limit_price=order.target_price),
        )
        sub = self._trading.submit_order(req)
        return str(sub.id)

    def submit_variant_entry(self, order: VariantOrder) -> str:
        ""
        from alpaca.trading.requests import (
            LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest,
            StopLossRequest, StopOrderRequest, TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        common = dict(symbol=order.symbol, qty=order.qty,
                      side=OrderSide.BUY, time_in_force=TimeInForce.GTC)

        if order.exit_type == "bracket" and order.entry_type in ("market_open", "limit"):
            legs = dict(
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=order.stop_price),
                take_profit=TakeProfitRequest(limit_price=order.target_price),
            )
            if order.entry_type == "market_open":
                req = MarketOrderRequest(**common, **legs)
            else:
                req = LimitOrderRequest(**common, **legs,
                                        limit_price=order.entry_price)
        elif order.entry_type == "market_open":
            req = MarketOrderRequest(**common)
        elif order.entry_type == "limit":
            req = LimitOrderRequest(**common, limit_price=order.entry_price)
        elif order.entry_type == "stop":
            req = StopOrderRequest(**common, stop_price=order.stop_trigger)
        elif order.entry_type == "stop_limit":
            req = StopLimitOrderRequest(**common, stop_price=order.stop_trigger,
                                        limit_price=order.entry_price)
        else:
            raise ValueError(f"unknown entry_type {order.entry_type!r}")
        sub = self._trading.submit_order(req)
        return str(sub.id)


    def submit_trailing_stop(self, symbol: str, qty: int, trail_pct: float) -> str:
        ""
        from alpaca.trading.requests import TrailingStopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = TrailingStopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=round(trail_pct * 100, 2),
        )
        return str(self._trading.submit_order(req).id)

    def submit_stop_sell(self, symbol: str, qty: int, stop_price: float) -> str:
        ""
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                               time_in_force=TimeInForce.GTC,
                               stop_price=stop_price)
        return str(self._trading.submit_order(req).id)

    def submit_limit_sell(self, symbol: str, qty: int, limit_price: float,
                          oco_with_stop: float | None = None) -> str:
        ""
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        kwargs = dict(symbol=symbol, qty=qty, side=OrderSide.SELL,
                      time_in_force=TimeInForce.GTC, limit_price=limit_price)
        if oco_with_stop is not None:
            kwargs["order_class"] = OrderClass.OCO
            kwargs["stop_loss"] = StopLossRequest(stop_price=oco_with_stop)
        return str(self._trading.submit_order(LimitOrderRequest(**kwargs)).id)

    def submit_moc_sell(self, symbol: str, qty: int) -> str:
        ""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                 time_in_force=TimeInForce.CLS)
        return str(self._trading.submit_order(req).id)


    def submit_market_bracket(self, symbol: str, qty: int,
                              stop_price: float, target_price: float) -> str:
        ""
        from alpaca.trading.requests import (
            MarketOrderRequest, StopLossRequest, TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC, order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        sub = self._trading.submit_order(req)
        return str(sub.id)

    def get_order_legs(self, order_id: str) -> list[dict]:
        ""
        o = self._trading.get_order_by_id(order_id)
        return [self._normalize_leg(lg) for lg in (getattr(o, "legs", None) or [])]

    @staticmethod
    def _normalize_leg(lg) -> dict:
        ltype = getattr(lg.type, "value", str(lg.type))
        stop_price = float(lg.stop_price) if getattr(lg, "stop_price", None) else None
        limit_price = float(lg.limit_price) if getattr(lg, "limit_price", None) else None
        if "stop" in ltype or stop_price is not None:
            kind = "stop"
        elif "limit" in ltype:
            kind = "target"
        else:
            kind = "other"
        return {
            "id": str(lg.id),
            "kind": kind,
            "type": ltype,
            "side": getattr(lg.side, "value", str(lg.side)),
            "status": getattr(lg.status, "value", str(lg.status)),
            "stop_price": stop_price,
            "limit_price": limit_price,
            "qty": int(float(lg.qty)) if getattr(lg, "qty", None) else None,
        }

    def submit_oco_exit(self, symbol: str, qty: int,
                        stop_price: float, target_price: float) -> dict:
        ""
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        req = LimitOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, order_class=OrderClass.OCO,
            limit_price=target_price,
            stop_loss=StopLossRequest(stop_price=stop_price),
        )
        sub = self._trading.submit_order(req)
        legs = [self._normalize_leg(lg) for lg in (getattr(sub, "legs", None) or [])]
        stop_id = next((l["id"] for l in legs if l["kind"] == "stop"), None)
        target_id = str(sub.id)
        primary = self._normalize_leg(sub) if hasattr(sub, "type") else None
        if primary and primary["kind"] == "stop":
            stop_id = str(sub.id)
            target_id = next((l["id"] for l in legs if l["kind"] == "target"), None)
        return {"oco_order_id": str(sub.id), "stop_order_id": stop_id,
                "target_order_id": target_id}

    def get_open_orders(self) -> list[dict]:
        ""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        out: list[dict] = []
        for o in self._trading.get_orders(req):
            out.append(self._flatten_order(o))
            for lg in (getattr(o, "legs", None) or []):
                leg = self._normalize_leg(lg)
                leg["symbol"] = getattr(lg, "symbol", o.symbol)
                leg["order_class"] = getattr(o.order_class, "value", str(o.order_class))
                out.append(leg)
        return out

    @staticmethod
    def _flatten_order(o) -> dict:
        return {
            "id": str(o.id),
            "symbol": o.symbol,
            "kind": AlpacaClient._normalize_leg(o)["kind"] if hasattr(o, "type") else "other",
            "type": getattr(o.type, "value", str(o.type)),
            "side": getattr(o.side, "value", str(o.side)),
            "status": getattr(o.status, "value", str(o.status)),
            "stop_price": float(o.stop_price) if getattr(o, "stop_price", None) else None,
            "limit_price": float(o.limit_price) if getattr(o, "limit_price", None) else None,
            "qty": int(float(o.qty)) if getattr(o, "qty", None) else None,
            "order_class": getattr(o.order_class, "value", str(o.order_class)),
        }

    def get_order(self, order_id: str) -> dict:
        ""
        o = self._trading.get_order_by_id(order_id)
        status = getattr(o.status, "value", str(o.status))
        return {
            "id": str(o.id),
            "status": status,
            "filled_qty": int(float(o.filled_qty or 0)),
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
            "filled_at": o.filled_at.isoformat() if getattr(o, "filled_at", None) else None,
        }

    def get_trading_days(self, start: str, end: str) -> list[str]:
        ""
        from datetime import date as _date
        from alpaca.trading.requests import GetCalendarRequest
        cal = self._trading.get_calendar(
            GetCalendarRequest(start=_date.fromisoformat(start), end=_date.fromisoformat(end)))
        return sorted(c.date.isoformat() if hasattr(c.date, "isoformat") else str(c.date)
                      for c in cal)

    def next_trading_day(self, after: str) -> str | None:
        ""
        from datetime import date as _date, timedelta
        d0 = _date.fromisoformat(after)
        days = self.get_trading_days((d0 + timedelta(days=1)).isoformat(),
                                     (d0 + timedelta(days=10)).isoformat())
        return days[0] if days else None

    def is_trading_day(self, day: str) -> bool:
        return day in self.get_trading_days(day, day)


class StubAlpacaClient(AlpacaClient):
    ""

    def __init__(self):
        self._fake_account = AccountSnapshot(
            cash=1000.0, equity=1000.0, buying_power=2000.0, portfolio_value=1000.0,
        )
        self._fake_positions: list[Position] = []

    def get_account(self) -> AccountSnapshot:
        return self._fake_account

    def get_positions(self) -> list[Position]:
        return list(self._fake_positions)

    def get_quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
        return {s: Quote(symbol=s, bid=100.0, ask=100.1, mid=100.05) for s in symbols}

    def submit_bracket(self, order: BracketOrder) -> str:
        return f"stub-order-{order.symbol}"

    def submit_variant_entry(self, order: VariantOrder) -> str:
        return f"stub-variant-{order.entry_type}-{order.symbol}"

    def submit_trailing_stop(self, symbol: str, qty: int, trail_pct: float) -> str:
        return f"stub-trail-{symbol}"

    def submit_stop_sell(self, symbol: str, qty: int, stop_price: float) -> str:
        return f"stub-stop-{symbol}"

    def submit_limit_sell(self, symbol: str, qty: int, limit_price: float,
                          oco_with_stop: float | None = None) -> str:
        return f"stub-limit-{symbol}"

    def submit_moc_sell(self, symbol: str, qty: int) -> str:
        return f"stub-moc-{symbol}"

    def submit_market_bracket(self, symbol: str, qty: int,
                              stop_price: float, target_price: float) -> str:
        return f"stub-market-bracket-{symbol}"

    def get_order(self, order_id: str) -> dict:
        return {"id": order_id, "status": "filled",
                "filled_qty": 0, "filled_avg_price": None, "filled_at": None}

    def get_order_legs(self, order_id: str) -> list[dict]:
        return []

    def get_open_orders(self) -> list[dict]:
        return []

    def submit_oco_exit(self, symbol: str, qty: int,
                        stop_price: float, target_price: float) -> dict:
        return {"oco_order_id": f"stub-oco-{symbol}",
                "stop_order_id": f"stub-stop-{symbol}",
                "target_order_id": f"stub-tgt-{symbol}"}

    def get_trading_days(self, start: str, end: str) -> list[str]:
        from datetime import date as _date, timedelta
        out, d = [], _date.fromisoformat(start)
        end_d = _date.fromisoformat(end)
        while d <= end_d:
            if d.weekday() < 5:
                out.append(d.isoformat())
            d += timedelta(days=1)
        return out

    def next_trading_day(self, after: str) -> str | None:
        from datetime import date as _date, timedelta
        d0 = _date.fromisoformat(after) + timedelta(days=1)
        for _ in range(10):
            if d0.weekday() < 5:
                return d0.isoformat()
            d0 += timedelta(days=1)
        return None

    def is_trading_day(self, day: str) -> bool:
        from datetime import date as _date
        return _date.fromisoformat(day).weekday() < 5
