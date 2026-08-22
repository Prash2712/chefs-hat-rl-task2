from __future__ import annotations

import os
from dataclasses import dataclass

import requests

BASE_URL = "https://api.company-information.service.gov.uk"


class CompaniesHouseError(RuntimeError):
    """Raised when Companies House returns an unsuccessful response."""


@dataclass
class CompaniesHouseClient:
    api_key: str | None = None
    timeout: int = 30

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("COMPANIES_HOUSE_API_KEY")

    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise CompaniesHouseError("COMPANIES_HOUSE_API_KEY is not configured")
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params,
            auth=(self.api_key, ""),
            timeout=self.timeout,
        )
        if response.status_code == 404:
            raise CompaniesHouseError("Company or resource was not found")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise CompaniesHouseError(
                f"Companies House request failed with HTTP {response.status_code}"
            ) from exc
        return response.json()

    def company_profile(self, company_number: str) -> dict:
        return self._get(f"/company/{company_number.strip().upper()}")

    def filing_history(self, company_number: str, items_per_page: int = 100) -> dict:
        return self._get(
            f"/company/{company_number.strip().upper()}/filing-history",
            params={"items_per_page": min(max(items_per_page, 1), 100)},
        )

    def officers(self, company_number: str, items_per_page: int = 100) -> dict:
        return self._get(
            f"/company/{company_number.strip().upper()}/officers",
            params={"items_per_page": min(max(items_per_page, 1), 100)},
        )
