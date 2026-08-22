# CompanyScope UK

Companies House already exposes a lot of useful information. The awkward part is that a quick company check usually means opening several resources and mentally joining them: status, accounts, confirmation statements, filings, officers and, sometimes, insolvency history.

This project turns those public records into one inspectable report.

I deliberately did **not** build an “AI credit score”. There is no labelled credit-performance dataset here, and a made-up probability would look more impressive than it is useful. The current assessment is a transparent review-attention rule set: every flag comes directly from a public field and every point has a reason.

## Source

The client uses the Companies House Public Data API:

```text
/company/{company_number}
/company/{company_number}/filing-history
/company/{company_number}/officers
```

API docs: https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference

Set the key through the environment:

```bash
export COMPANIES_HOUSE_API_KEY='...'
```

Nothing secret is stored in source control.

## What the report contains

The feature layer currently derives:

- company status and type
- incorporation date / company age
- accounts-overdue flag
- confirmation-statement-overdue flag
- most recent filing date
- days since the latest filing
- filing count over the previous 12 months
- active officer count
- SIC codes
- whether Companies House exposes an insolvency-history resource

These are deterministic transformations. I want it to be possible to trace a value in the API response back to the source record without reverse-engineering a model.

## Review-attention rules

A few conditions add review points: non-active status, overdue statutory filings, very stale filing activity and an insolvency-history resource. The API returns the score, band and the exact reasons.

The bands are only:

```text
routine
review
high-attention
```

They are not a statement that a company is safe, unsafe, fraudulent, insolvent or creditworthy. Someone using the output still has to inspect the underlying records and context.

The rationale and current weights are documented in `docs/methodology.md`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

export COMPANIES_HOUSE_API_KEY='your_key_here'
companyscope inspect 00000006
```

The CLI is useful when I want the raw consolidated report without running a service.

For the API:

```bash
uvicorn companyscope.api:app --reload
```

Endpoints:

- `GET /health`
- `GET /company/{company_number}`

A Dockerfile is included for the same API path.

## Code map

```text
src/companyscope/client.py       Companies House requests
src/companyscope/features.py     deterministic feature building
src/companyscope/assessment.py   review-attention rules
src/companyscope/service.py      orchestration
src/companyscope/api.py          HTTP interface
src/companyscope/cli.py          command-line interface
tests/                           synthetic API-shaped fixtures
docs/methodology.md              assumptions and boundaries
```

Tests do not call the live Companies House service. They use small shaped payloads so a temporary API problem cannot make CI red.

```bash
ruff check src tests
pytest -q
```

## If I turn this into a predictive project

I would first build a **point-in-time** dataset. Using today's company status to predict an event that happened in the past would leak the answer immediately. The label also needs a precise definition and observation window. Until that data work exists, keeping the rules transparent is the more defensible design.

## One limitation worth calling out

A company can file on time and still be a poor counterparty; it can also have unusual filing activity for completely legitimate reasons. Public-record screening narrows what to inspect. It does not replace financial analysis or due diligence.

**Prasanth Balisetty**