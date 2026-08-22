from __future__ import annotations

from fastapi import FastAPI, HTTPException

from companyscope.client import CompaniesHouseError
from companyscope.service import build_company_report

app = FastAPI(
    title="CompanyScope UK",
    version="0.1.0",
    description="Companies House corporate intelligence and governance-screening API",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/company/{company_number}")
def company(company_number: str) -> dict:
    try:
        return build_company_report(company_number)
    except CompaniesHouseError as exc:
        detail = str(exc)
        status = 503 if "not configured" in detail else 502
        if "not found" in detail.lower():
            status = 404
        raise HTTPException(status_code=status, detail=detail) from exc
