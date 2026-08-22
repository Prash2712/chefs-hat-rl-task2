# Methodology and Governance Boundaries

## Purpose

CompanyScope turns Companies House public records into a compact due-diligence view. The system is intentionally designed for **triage and investigation**, not automated lending, insurance, employment or investment decisions.

## Source records

The current service uses three Companies House Public Data API resources:

1. Company profile
2. Filing history
3. Officers

The application does not scrape company websites or infer private information.

## Derived features

The feature layer exposes only interpretable transformations, including:

- company age
- company status
- accounts-overdue flag
- confirmation-statement-overdue flag
- days since latest filing
- filing count in the last 12 months
- active-officer count
- SIC codes
- presence of a Companies House insolvency-history resource

## Attention assessment

The score is an explicit rule set rather than a trained black-box model:

| Signal | Points |
|---|---:|
| status other than active | +35 |
| accounts overdue | +30 |
| confirmation statement overdue | +20 |
| no filing for >24 months | +10 |
| no filing for >18 months | +5 |
| insolvency-history resource exposed | +5 |

Bands:

- 0–29: `routine`
- 30–59: `review`
- 60–100: `high-attention`

Each response includes the exact reasons that contributed points.

## What the score is not

It is not:

- a credit rating
- a probability of insolvency
- a fraud score
- a recommendation to transact or refuse service
- a substitute for Companies House source records or professional due diligence

The labels are deliberately phrased as **attention bands** to avoid implying predictive validity that has not been established.

## Production extensions

A production-grade corporate-intelligence system could add:

- point-in-time snapshots for historical backtesting
- director-network analysis
- charges and insolvency endpoints
- sector peer benchmarking
- feature freshness monitoring
- entity resolution
- audited outcome labels before any predictive distress model is attempted

Any supervised risk model should be trained only after a defensible outcome definition, point-in-time leakage controls, temporal validation and fairness/governance review are established.
