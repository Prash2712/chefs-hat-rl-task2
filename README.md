# CompanyScope UK — Corporate Intelligence from Companies House

A production-style data product that turns **Companies House public records** into an explainable company due-diligence view: statutory filing signals, company age, filing recency, active-officer counts and a transparent **review-attention assessment**.

The project is intentionally built as a real service rather than a notebook. It has an authenticated source client, feature layer, explainable rules, FastAPI interface, CLI, tests, Docker and CI.

## The problem

Company information is public, but useful investigation often requires joining several resources and converting raw records into consistent signals. CompanyScope answers questions such as:

- Is the company active?
- Are accounts or confirmation statements marked overdue?
- When was the latest filing?
- How active has filing been in the past year?
- How many officers are currently active?
- Which observable public-record conditions deserve human review?

The system **does not claim to predict insolvency or creditworthiness**. It produces a traceable screening indicator for prioritising further investigation.

## Official data source

CompanyScope uses the **Companies House Public Data API**:

- Company profile: `/company/{company_number}`
- Filing history: `/company/{company_number}/filing-history`
- Officers: `/company/{company_number}/officers`

Documentation:

https://developer-specs.company-information.service.gov.uk/companies-house-public-data-api/reference

A Companies House API key is required. It is provided through the environment and never committed to source control.

## Architecture

```text
Companies House Public Data API
             |
             v
Authenticated API client
             |
       +-----+---------+
       |       |       |
       v       v       v
    profile  filings  officers
       \       |       /
        \      |      /
         v     v     v
       feature layer
             |
             v
transparent attention rules
             |
        +----+----+
        |         |
        v         v
      FastAPI     CLI
```

## Derived analytical features

The current feature layer exposes:

- company status and type
- date of creation / company age
- accounts-overdue flag
- confirmation-statement-overdue flag
- latest filing date
- days since latest filing
- filings in the past 12 months
- active officer count
- SIC codes
- presence of a Companies House insolvency-history resource

These are deterministic transformations of source fields, not inferred private attributes.

## Explainable attention assessment

The assessment is a small, auditable rule set. Every point is returned with a human-readable reason.

Examples of configured attention signals:

- company status other than active
- overdue accounts
- overdue confirmation statement
- unusually stale filing history
- an insolvency-history resource being exposed by Companies House

Bands are labelled `routine`, `review` and `high-attention`.

**Important:** this is not a credit rating, fraud score, probability of insolvency, investment recommendation or automated decision system. See [`docs/methodology.md`](docs/methodology.md).

## Repository structure

```text
.
├── src/companyscope/
│   ├── client.py
│   ├── features.py
│   ├── assessment.py
│   ├── service.py
│   ├── api.py
│   └── cli.py
├── tests/
├── docs/
│   └── methodology.md
├── .github/workflows/ci.yml
├── .env.example
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

export COMPANIES_HOUSE_API_KEY='your_key_here'
companyscope inspect 00000006
```

The CLI prints the company identity, derived features, attention band, score, exact contributing reasons and source label.

## API

```bash
uvicorn companyscope.api:app --reload
```

Endpoints:

- `GET /health`
- `GET /company/{company_number}`

The company endpoint fetches the live public record and returns the consolidated intelligence report.

## Container

```bash
docker build -t companyscope-uk .
docker run \
  -e COMPANIES_HOUSE_API_KEY="$COMPANIES_HOUSE_API_KEY" \
  -p 8000:8000 \
  companyscope-uk
```

## Quality controls

Tests use synthetic Companies House-shaped payloads and do not depend on live API responses.

```bash
ruff check src tests
pytest -q
```

GitHub Actions runs both checks on pushes and pull requests.

## Why this is a portfolio data product

**Data / Analytics**
- multi-resource public-record integration
- explicit feature definitions
- reproducible business rules
- data freshness measures

**Software / ML Engineering**
- authenticated external API client
- environment-based secret handling
- package architecture
- service composition
- FastAPI
- Docker
- CI and unit testing

**Model governance**
- no unsupported predictive claims
- transparent score components
- explicit intended-use boundary
- pathway documented for future point-in-time supervised modelling

## Suggested repository name

Rename this current legacy shell to:

`uk-company-intelligence-platform`

The original reinforcement-learning coursework is preserved on `archive/original-coursework`.

## Author

**Prasanth Balisetty**  
Data Science · Analytics · Machine Learning Engineering
