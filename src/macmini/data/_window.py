""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SIP_RECENT_BUFFER = timedelta(minutes=16)

SESSION_FINAL_HOUR = 16
SESSION_FINAL_MINUTE = 16


def clamp_feed_end(end: datetime, now: datetime | None = None) -> datetime:
    ""
    if end.tzinfo is None:
        raise ValueError("clamp_feed_end requires a tz-aware end datetime")
    now = now or datetime.now(timezone.utc)
    clamp = now - SIP_RECENT_BUFFER
    et_now = now.astimezone(ET)
    if (et_now.hour, et_now.minute) < (SESSION_FINAL_HOUR, SESSION_FINAL_MINUTE):
        midnight_et = datetime(
            et_now.year, et_now.month, et_now.day, tzinfo=ET
        ).astimezone(timezone.utc)
        clamp = min(clamp, midnight_et)
    return min(end, clamp)
