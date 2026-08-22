from companyscope.service import build_company_report


class FakeClient:
    def company_profile(self, company_number: str) -> dict:
        return {
            "company_number": company_number,
            "company_name": "EXAMPLE DATA LTD",
            "company_status": "active",
            "type": "ltd",
            "date_of_creation": "2019-06-01",
            "accounts": {"overdue": False},
            "confirmation_statement": {"overdue": False},
            "sic_codes": ["62012"],
            "links": {},
        }

    def filing_history(self, company_number: str) -> dict:
        return {"items": [{"date": "2026-07-01", "category": "accounts"}]}

    def officers(self, company_number: str) -> dict:
        return {"items": [{"name": "Director One"}]}


def test_build_company_report_is_explainable():
    report = build_company_report("01234567", client=FakeClient())

    assert report["company"]["company_name"] == "EXAMPLE DATA LTD"
    assert report["attention_assessment"]["score"] == 0
    assert report["attention_assessment"]["band"] == "routine"
    assert report["source"] == "Companies House Public Data API"
