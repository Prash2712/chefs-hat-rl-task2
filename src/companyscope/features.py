from __future__ import annotations

from datetime import UTC, date, datetime


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _days_between(earlier: date | None, later: date) -> int | None:
    return None if earlier is None else (later - earlier).days


def build_company_features(
    profile: dict,
    filing_history: dict,
    officers: dict,
    as_of: date | None = None,
) -> dict[str, object]:
    """Create auditable features from Companies House public-record responses."""
    as_of = as_of or datetime.now(UTC).date()
    creation_date = _parse_date(profile.get("date_of_creation"))
    company_age_days = _days_between(creation_date, as_of)

    filings = filing_history.get("items", []) or []
    filing_dates = [
        parsed
        for item in filings
        if (parsed := _parse_date(item.get("date"))) is not None
    ]
    latest_filing_date = max(filing_dates) if filing_dates else None
    recent_12m_cutoff_days = 365
    filings_last_12m = sum(
        1 for filing_date in filing_dates if (as_of - filing_date).days <= recent_12m_cutoff_days
    )

    officer_items = officers.get("items", []) or []
    active_officers = sum(1 for officer in officer_items if not officer.get("resigned_on"))

    accounts = profile.get("accounts", {}) or {}
    confirmation = profile.get("confirmation_statement", {}) or {}

    return {
        "company_number": profile.get("company_number"),
        "company_name": profile.get("company_name"),
        "company_status": profile.get("company_status"),
        "company_type": profile.get("type"),
        "date_of_creation": creation_date.isoformat() if creation_date else None,
        "company_age_days": company_age_days,
        "accounts_overdue": bool(accounts.get("overdue", False)),
        "confirmation_statement_overdue": bool(confirmation.get("overdue", False)),
        "latest_filing_date": latest_filing_date.isoformat() if latest_filing_date else None,
        "days_since_latest_filing": _days_between(latest_filing_date, as_of),
        "filings_last_12m": filings_last_12m,
        "active_officer_count": active_officers,
        "sic_codes": profile.get("sic_codes", []) or [],
        "has_insolvency_history_link": bool((profile.get("links", {}) or {}).get("insolvency")),
    }
