""
from datetime import datetime, timedelta, timezone

from macmini.data._window import ET, clamp_feed_end


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


class TestClampFeedEnd:
    def test_evening_chain_future_end_is_pulled_inside_sip_window(self):
        now = _et(2026, 8, 28, 19, 50).astimezone(timezone.utc)
        end = _utc(2026, 8, 29, 0, 0)
        clamped = clamp_feed_end(end, now=now)
        assert clamped == now - timedelta(minutes=16)
        assert clamped > _et(2026, 8, 28, 16, 0).astimezone(timezone.utc)

    def test_intraday_run_excludes_developing_session(self):
        now = _et(2026, 9, 1, 10, 0).astimezone(timezone.utc)
        end = _utc(2026, 9, 2, 0, 0)
        clamped = clamp_feed_end(end, now=now)
        assert clamped == _et(2026, 9, 1, 0, 0).astimezone(timezone.utc)

    def test_premarket_run_includes_yesterday(self):
        now = _et(2026, 9, 1, 7, 30).astimezone(timezone.utc)
        clamped = clamp_feed_end(_utc(2026, 9, 2, 0, 0), now=now)
        assert clamped == _et(2026, 9, 1, 0, 0).astimezone(timezone.utc)
        assert clamped > _et(2026, 8, 31, 16, 0).astimezone(timezone.utc)

    def test_post_settle_run_includes_today(self):
        now = _et(2026, 9, 1, 16, 20).astimezone(timezone.utc)
        clamped = clamp_feed_end(_utc(2026, 9, 2, 0, 0), now=now)
        assert clamped == now - timedelta(minutes=16)

    def test_historical_end_untouched(self):
        now = _et(2026, 9, 1, 10, 0).astimezone(timezone.utc)
        end = _utc(2026, 8, 21, 0, 0)
        assert clamp_feed_end(end, now=now) == end

    def test_naive_end_rejected(self):
        try:
            clamp_feed_end(datetime(2026, 9, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("naive datetime must be rejected")
