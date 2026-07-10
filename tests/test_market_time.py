"""Pins the NYSE calendar and session-arithmetic contract.

The calendar is *computed* from exchange rules (observation shifts,
nth-weekday rules, the Easter computus), so these tests check concrete,
externally verifiable 2026 dates: if a rule regresses, a named holiday
moves and a test here names the failure.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from aether.utils import market_time as mt


# --------------------------------------------------------------------------- #
# Full-day holidays
# --------------------------------------------------------------------------- #

class TestHolidays:
    def test_independence_day_2026_observed_on_july_3(self) -> None:
        # July 4 2026 is a Saturday -> NYSE observes Friday July 3.
        assert date(2026, 7, 4).weekday() == 5
        assert date(2026, 7, 3) in mt.nyse_holidays(2026)
        assert not mt.is_trading_day(date(2026, 7, 3))
        # The following Monday is an ordinary trading day.
        assert mt.is_trading_day(date(2026, 7, 6))

    def test_good_friday_2026(self) -> None:
        # Easter 2026 falls on April 5 -> Good Friday is April 3.
        assert date(2026, 4, 3) in mt.nyse_holidays(2026)
        assert not mt.is_trading_day(date(2026, 4, 3))
        # Maundy Thursday trades normally.
        assert mt.is_trading_day(date(2026, 4, 2))

    def test_fixed_and_floating_holidays_2026(self) -> None:
        hol = mt.nyse_holidays(2026)
        assert date(2026, 1, 1) in hol      # New Year's Day (Thursday)
        assert date(2026, 1, 19) in hol     # MLK: 3rd Monday of January
        assert date(2026, 11, 26) in hol    # Thanksgiving: 4th Thursday of Nov
        assert date(2026, 12, 25) in hol    # Christmas (Friday)

    def test_weekends_are_never_trading_days(self) -> None:
        assert not mt.is_trading_day(date(2026, 1, 10))  # Saturday
        assert not mt.is_trading_day(date(2026, 1, 11))  # Sunday


# --------------------------------------------------------------------------- #
# Half days (early 13:00 close)
# --------------------------------------------------------------------------- #

class TestHalfDays:
    def test_black_friday_2026_is_a_210_minute_session(self) -> None:
        d = date(2026, 11, 27)              # day after Thanksgiving
        assert d in mt.nyse_half_days(2026)
        assert mt.session_close(d) == time(13, 0)
        assert mt.expected_session_minutes(d) == 210

    def test_full_day_has_390_minutes(self) -> None:
        d = date(2026, 1, 5)                # ordinary Monday
        assert mt.is_trading_day(d)
        assert d not in mt.nyse_half_days(2026)
        assert mt.session_close(d) == time(16, 0)
        assert mt.expected_session_minutes(d) == 390

    def test_half_days_are_still_trading_days(self) -> None:
        assert mt.is_trading_day(date(2026, 11, 27))
        assert date(2026, 11, 27) in mt.trading_days(date(2026, 11, 27),
                                                     date(2026, 11, 27))


# --------------------------------------------------------------------------- #
# session_minute boundaries
# --------------------------------------------------------------------------- #

class TestSessionMinute:
    def test_open_is_minute_zero(self) -> None:
        assert mt.session_minute(datetime(2026, 1, 5, 9, 30)) == 0

    def test_last_bar_is_minute_389(self) -> None:
        assert mt.session_minute(datetime(2026, 1, 5, 15, 59)) == 389

    def test_close_raises(self) -> None:
        # 16:00 is the exclusive close: there is no 1min bar stamped 16:00.
        with pytest.raises(ValueError):
            mt.session_minute(datetime(2026, 1, 5, 16, 0))

    def test_premarket_raises(self) -> None:
        with pytest.raises(ValueError):
            mt.session_minute(datetime(2026, 1, 5, 9, 29))

    def test_timezone_aware_input_converted_to_eastern(self) -> None:
        # 14:30 UTC == 09:30 ET during EST (winter, UTC-5).
        aware = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        assert mt.session_minute(aware) == 0

    def test_midday_minute(self) -> None:
        # 12:00 is 2.5 hours = 150 minutes after the open.
        assert mt.session_minute(datetime(2026, 1, 5, 12, 0)) == 150


# --------------------------------------------------------------------------- #
# in_regular_session
# --------------------------------------------------------------------------- #

class TestInRegularSession:
    def test_regular_hours(self) -> None:
        assert mt.in_regular_session(datetime(2026, 1, 5, 9, 30))
        assert mt.in_regular_session(datetime(2026, 1, 5, 15, 59))

    def test_outside_hours(self) -> None:
        assert not mt.in_regular_session(datetime(2026, 1, 5, 9, 29))
        assert not mt.in_regular_session(datetime(2026, 1, 5, 16, 0))

    def test_holiday_and_weekend(self) -> None:
        assert not mt.in_regular_session(datetime(2026, 1, 19, 12, 0))  # MLK
        assert not mt.in_regular_session(datetime(2026, 1, 10, 12, 0))  # Sat

    def test_half_day_close_respected(self) -> None:
        # Black Friday 2026 closes 13:00: 12:59 is in-session, 13:00 is not.
        assert mt.in_regular_session(datetime(2026, 11, 27, 12, 59))
        assert not mt.in_regular_session(datetime(2026, 11, 27, 13, 0))


# --------------------------------------------------------------------------- #
# trading_days
# --------------------------------------------------------------------------- #

class TestTradingDays:
    def test_january_2026(self) -> None:
        days = mt.trading_days(date(2026, 1, 1), date(2026, 1, 31))
        # 22 weekdays in Jan 2026 minus New Year's Day and MLK Day.
        assert len(days) == 20
        assert all(d.weekday() < 5 for d in days)
        assert date(2026, 1, 1) not in days     # New Year's Day
        assert date(2026, 1, 19) not in days    # MLK Day
        assert days[0] == date(2026, 1, 2)      # first session of the year
        assert date(2026, 1, 5) in days

    def test_sorted_and_inclusive(self) -> None:
        days = mt.trading_days(date(2026, 1, 5), date(2026, 1, 12))
        assert days == sorted(days)
        assert days[0] == date(2026, 1, 5)
        assert days[-1] == date(2026, 1, 12)
        # Mon 5 .. Fri 9 plus Mon 12; the weekend 10/11 is skipped.
        assert len(days) == 6

    def test_observed_holiday_excluded(self) -> None:
        days = mt.trading_days(date(2026, 7, 1), date(2026, 7, 10))
        assert date(2026, 7, 3) not in days
        assert date(2026, 7, 2) in days
        assert date(2026, 7, 6) in days
