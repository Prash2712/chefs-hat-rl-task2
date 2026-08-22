from datetime import date

from companyscope.features import build_company_features


def test_build_company_features_from_public_record_payloads():
    profile = {
        "company_number": "01234567",
        "company_name": "EXAMPLE ANALYTICS LTD",
        "company_status": "active",
        "type": "ltd",
        "date_of_creation": "2020-01-15",
        "sic_codes": ["62020"],
        "accounts": {"overdue": True},
        "confirmation_statement": {"overdue": False},
        "links": {"self": "/company/01234567"},
    }
    filings = {
        "items": [
            {"date": "2026-02-01", "category": "accounts"},
            {"date": "2025-09-01", "category": "confirmation-statement"},
        ]
    }
    officers = {
        "items": [
            {"name": "A Director"},
            {"name": "Former Director", "resigned_on": "2025-01-01"},
        ]
    }

    features = build_company_features(profile, filings, officers, as_of=date(2026, 8, 22))

    assert features["company_number"] == "01234567"
    assert features["accounts_overdue"] is True
    assert features["confirmation_statement_overdue"] is False
    assert features["active_officer_count"] == 1
    assert features["filings_last_12m"] == 2
    assert features["latest_filing_date"] == "2026-02-01"
    assert features["days_since_latest_filing"] == 202
