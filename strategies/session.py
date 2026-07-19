"""
session.py -- Market session time helpers.
"""
from calendar import monthrange
from datetime import date, datetime, time, timedelta

import pytz

import config


def now_eastern() -> datetime:
    tz = pytz.timezone(config.TIMEZONE)
    return datetime.now(tz)


def current_session_time() -> time:
    return now_eastern().time()


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    month_offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * month_offset) // 451
    month = (h + month_offset - 7 * m + 114) // 31
    day = (h + month_offset - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nyse_holidays(year: int) -> set[date]:
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    return holidays


def is_market_day(day: date = None) -> bool:
    day = day or now_eastern().date()
    if day.weekday() >= 5:
        return False
    holidays = _nyse_holidays(day.year) | _nyse_holidays(day.year + 1)
    return day not in holidays


def is_premarket() -> bool:
    if not is_market_day():
        return False
    t = current_session_time()
    return config.PREMARKET_START <= t < config.MARKET_OPEN


def is_market_open() -> bool:
    if not is_market_day():
        return False
    t = current_session_time()
    return config.MARKET_OPEN <= t < config.MARKET_CLOSE


def is_postmarket() -> bool:
    if not is_market_day():
        return False
    t = current_session_time()
    return config.MARKET_CLOSE <= t < config.POSTMARKET_END


def is_trading_hours() -> bool:
    return is_market_open()


def minutes_until_close() -> float:
    now = now_eastern()
    close_dt = now.replace(
        hour=config.MARKET_CLOSE.hour,
        minute=config.MARKET_CLOSE.minute,
        second=0, microsecond=0,
    )
    delta = (close_dt - now).total_seconds() / 60.0
    return max(delta, 0.0)


def should_flatten() -> bool:
    return is_market_open() and minutes_until_close() <= config.FLATTEN_BEFORE_CLOSE_MINUTES


def should_warn_flatten() -> bool:
    return is_market_open() and minutes_until_close() <= config.FLATTEN_WARNING_MINUTES


def session_status() -> str:
    if is_premarket():
        return "PREMARKET"
    if is_market_open():
        if should_flatten():
            return "FLATTEN_ZONE"
        if should_warn_flatten():
            return "CLOSE_WARNING"
        return "OPEN"
    if is_postmarket():
        return "POSTMARKET"
    return "CLOSED"


def orb_end_time() -> time:
    minutes = config.MARKET_OPEN.hour * 60 + config.MARKET_OPEN.minute + config.ORB_WINDOW_MINUTES
    return time(minutes // 60, minutes % 60)


def is_orb_window() -> bool:
    t = current_session_time()
    return config.MARKET_OPEN <= t < orb_end_time()
