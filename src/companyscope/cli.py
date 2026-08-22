from __future__ import annotations

import json

import typer

from companyscope.client import CompaniesHouseError
from companyscope.service import build_company_report

app = typer.Typer(help="CompanyScope UK public-record intelligence")


@app.command()
def inspect(company_number: str) -> None:
    """Fetch and print an explainable company intelligence report."""
    try:
        report = build_company_report(company_number)
    except CompaniesHouseError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
