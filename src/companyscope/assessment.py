from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AttentionAssessment:
    score: int
    band: str
    reasons: list[str]
    disclaimer: str = (
        "Screening indicator based on public filing/governance signals; "
        "not a credit rating, insolvency prediction or recommendation."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def assess_attention(features: dict[str, object]) -> AttentionAssessment:
    """Score transparent public-record flags; higher means more review attention."""
    score = 0
    reasons: list[str] = []

    status = str(features.get("company_status") or "").lower()
    if status and status != "active":
        score += 35
        reasons.append(f"Company status is '{status}' rather than active (+35)")

    if bool(features.get("accounts_overdue")):
        score += 30
        reasons.append("Accounts are marked overdue (+30)")

    if bool(features.get("confirmation_statement_overdue")):
        score += 20
        reasons.append("Confirmation statement is marked overdue (+20)")

    days_since_latest = features.get("days_since_latest_filing")
    if isinstance(days_since_latest, int) and days_since_latest > 730:
        score += 10
        reasons.append("No filing recorded in more than 24 months (+10)")
    elif isinstance(days_since_latest, int) and days_since_latest > 548:
        score += 5
        reasons.append("No filing recorded in more than 18 months (+5)")

    if bool(features.get("has_insolvency_history_link")):
        score += 5
        reasons.append("Companies House exposes an insolvency-history resource (+5)")

    score = min(score, 100)
    if score >= 60:
        band = "high-attention"
    elif score >= 30:
        band = "review"
    else:
        band = "routine"

    if not reasons:
        reasons.append("No configured attention flags were triggered")

    return AttentionAssessment(score=score, band=band, reasons=reasons)
