"""Deterministic publication clocks for the next evidence-packet version."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from news_event_calendar_2025 import _sessions

NY = ZoneInfo("America/New_York")


def aware_timestamp(value):
    stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)


def publication_clock(published_at, asof, forecast_sessions):
    published, decision = aware_timestamp(published_at), aware_timestamp(asof)
    if published > decision:
        raise ValueError("Future publication cannot enter evidence")
    local = published.astimezone(NY)
    origin = decision.astimezone(NY).date().isoformat()
    schedule = _sessions()
    origin_row = next((row for row in schedule if row[0] == origin), None)
    day_row = next((row for row in schedule if row[0] == local.date().isoformat()), None)
    first_close = next((row for row in schedule if row[2] > published), None)
    if not origin_row or not first_close or published.date().isoformat() < schedule[0][0]:
        raise ValueError("Exchange calendar coverage unavailable")
    if day_row is None:
        phase = "non_trading_day"
    elif published < day_row[1]:
        phase = "before_open"
    elif published < day_row[2]:
        phase = "during_session"
    else:
        phase = "at_or_after_close"
    completed = [row[0] for row in schedule if published < row[2] <= decision]
    return {
        "published_new_york": local.isoformat(), "publication_market_phase": phase,
        "publication_day_close_new_york": day_row[2].astimezone(NY).isoformat() if day_row else None,
        "decision_new_york": decision.astimezone(NY).isoformat(),
        "decision_day_exchange_close_new_york": origin_row[2].astimezone(NY).isoformat(),
        "publication_age_hours_at_decision": (decision-published).total_seconds()/3600,
        "first_return_session_after_publication": first_close[0],
        "publication_return_completed_by_decision": first_close[2] <= decision,
        "publication_return_in_forecast_window": first_close[0] in forecast_sessions,
        "completed_return_sessions_since_publication": completed,
        "completed_sessions_after_first_publication_return": completed[1:] if completed else [],
        "scope": "Publication timing only. A scheduled future event mentioned in an article has a separate event-calendar clock. Price discovery or uncertainty resolution cannot be inferred from this clock."
    }
