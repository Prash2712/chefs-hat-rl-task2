from __future__ import annotations

from companyscope.assessment import assess_attention
from companyscope.client import CompaniesHouseClient
from companyscope.features import build_company_features


def build_company_report(company_number: str, client: CompaniesHouseClient | None = None) -> dict:
    """Fetch public records and return a compact, explainable intelligence report."""
    client = client or CompaniesHouseClient()
    profile = client.company_profile(company_number)
    filings = client.filing_history(company_number)
    officers = client.officers(company_number)
    features = build_company_features(profile, filings, officers)
    assessment = assess_attention(features)
    return {
        "company": {
            "company_number": features["company_number"],
            "company_name": features["company_name"],
            "company_status": features["company_status"],
            "company_type": features["company_type"],
            "sic_codes": features["sic_codes"],
        },
        "features": features,
        "attention_assessment": assessment.to_dict(),
        "source": "Companies House Public Data API",
    }
