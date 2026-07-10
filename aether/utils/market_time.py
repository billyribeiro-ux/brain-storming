"""US equity market session calendar and time arithmetic.

Aether treats the regular NYSE/Nasdaq session (09:30–16:00 America/New_York)
as the canonical intraday clock: each session has 390 one-minute slots
(0..389) and 78 five-minute slots. All FMP intraday timestamps are quoted in
US/Eastern local time, which is why this module works in that zone.

The holiday calendar is computed from the exchange's published rules (fixed
dates with weekend observation shifts, nth-weekday rules, and Good Friday via
the Easter computus) rather than a hardcoded year list, so it stays correct
for any backfill range without maintenance. Early-close half days are listed
separately; downstream consumers treat short sessions tolerantly (data-driven)
and use this only for gap *auditing*, never for trading decisions.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)          # exclusive: last 1min bar is 15:59
MINUTES_PER_SESSION = 390
BARS_5MIN_PER_SESSION = 78


# --------------------------------------------------------------------------- #
# Holiday rules
# --------------------------------------------------------------------------- #

def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus (Meeus/Jones/Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date | None:
    """NYSE observation shift for fixed-date holidays.

    Saturday holidays are observed the preceding Friday; Sunday holidays the
    following Monday. A Saturday New Year's Day is NOT observed on the prior
    Friday (that Friday belongs to the old year), matching NYSE practice —
    the exchange simply stays open.
    """
    if d.weekday() == 5:  # Saturday
        return None if (d.month == 1 and d.day == 1) else d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


#: One-off full-day closures that no rule can predict (proclamations,
#: catastrophes). Verified against the lake: 2025-01-09 was the National Day
#: of Mourning for President Carter — every US equity venue closed. Extend
#: this set when history demands it.
SPECIAL_CLOSURES: frozenset[date] = frozenset({
    date(2025, 1, 9),
})


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset[date]:
    """Full-day market closures for a given year (rules + special closures)."""
    days: set[date] = {d for d in SPECIAL_CLOSURES if d.year == year}

    for fixed in (
        date(year, 1, 1),                              # New Year's Day
        date(year, 6, 19) if year >= 2022 else None,   # Juneteenth (from 2022)
        date(year, 7, 4),                              # Independence Day
        date(year, 12, 25),                            # Christmas Day
    ):
        if fixed is not None:
            obs = _observed(fixed)
            if obs is not None:
                days.add(obs)

    days.add(_nth_weekday(year, 1, 0, 3))              # MLK Day: 3rd Mon Jan
    days.add(_nth_weekday(year, 2, 0, 3))              # Presidents' Day: 3rd Mon Feb
    days.add(_easter_sunday(year) - timedelta(days=2)) # Good Friday
    days.add(_last_weekday(year, 5, 0))                # Memorial Day: last Mon May
    days.add(_nth_weekday(year, 9, 0, 1))              # Labor Day: 1st Mon Sep
    days.add(_nth_weekday(year, 11, 3, 4))             # Thanksgiving: 4th Thu Nov
    return frozenset(days)


@lru_cache(maxsize=64)
def nyse_half_days(year: int) -> frozenset[date]:
    """Early-close (13:00 ET) sessions: day before July 4 (when a weekday
    and the 4th is not observed elsewhere), day after Thanksgiving, and
    Christmas Eve when it falls on a weekday."""
    days: set[date] = set()
    july3 = date(year, 7, 3)
    if july3.weekday() < 5 and date(year, 7, 4).weekday() != 5:
        days.add(july3)
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))  # Black Friday
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5:
        days.add(xmas_eve)
    return frozenset(d for d in days if d not in nyse_holidays(year))


# --------------------------------------------------------------------------- #
# Session arithmetic
# --------------------------------------------------------------------------- #

def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def trading_days(start: date, end: date) -> list[date]:
    """All full or half trading sessions in [start, end], inclusive."""
    out, d = [], start
    while d <= end:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def session_close(d: date) -> time:
    return time(13, 0) if d in nyse_half_days(d.year) else SESSION_CLOSE


def expected_session_minutes(d: date) -> int:
    """Number of 1-minute bars a complete session should contain."""
    close = session_close(d)
    return (close.hour - SESSION_OPEN.hour) * 60 + (close.minute - SESSION_OPEN.minute)


def in_regular_session(ts: datetime) -> bool:
    """True if a (naive-ET or aware) timestamp falls inside regular hours."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(ET).replace(tzinfo=None)
    if not is_trading_day(ts.date()):
        return False
    return SESSION_OPEN <= ts.time() < session_close(ts.date())


def session_minute(ts: datetime) -> int:
    """Minutes elapsed since 09:30 ET (0..389). Raises outside the session."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(ET).replace(tzinfo=None)
    minute = (ts.hour - SESSION_OPEN.hour) * 60 + (ts.minute - SESSION_OPEN.minute)
    if not 0 <= minute < MINUTES_PER_SESSION:
        raise ValueError(f"{ts} is outside the regular session")
    return minute
