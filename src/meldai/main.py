"""MeldAI CLI and Application Entrypoint."""

import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
import uvicorn
from fastapi import FastAPI

from meldai.config import get_settings
from meldai.db.postgres import PostgresSource
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.sapbert import SapBERTEmbedder
from meldai.pipelines.medical_analysis import MedicalAnalysisPipeline
from meldai.api.router import api_router

import sys

settings = get_settings()
log_level = getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("meldai")
logger.setLevel(log_level)

app = typer.Typer(
    name="meldai",
    help="MeldAI: Clinical Intelligence — SapBERT Embeddings, HPO & MONDO Terminology",
    add_completion=False,
)
console = Console()


def create_web_app() -> FastAPI:
    """Create and configure FastAPI web application instance."""
    st = get_settings()
    lvl = getattr(logging, (st.log_level or "INFO").upper(), logging.INFO)
    logging.getLogger().setLevel(lvl)
    logging.getLogger("meldai").setLevel(lvl)

    web_app = FastAPI(
        title="MeldAI Clinical Intelligence API",
        description="REST API for medical data analysis, SapBERT embeddings, HPO and MONDO terminology",
        version="0.2.0",
    )
    web_app.include_router(api_router)
    return web_app


@app.command()
def check_health():
    """Verify connectivity to PostgreSQL and MongoDB, and show ontology config."""
    console.print(Panel.fit("[bold blue]🩺 MeldAI Infrastructure Health Check[/bold blue]"))
    settings = get_settings()

    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)

    pg_ok = pg.ping()
    mongo_ok = mongo.ping()

    table = Table(title="Service Connectivity Status", show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Target", style="white")
    table.add_column("Status", style="bold")

    table.add_row(
        "PostgreSQL (External Source)",
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
        "[green]ONLINE[/green]" if pg_ok else "[yellow]OFFLINE (Configure .env for external PG)[/yellow]",
    )
    table.add_row(
        "MongoDB (Knowledge Base Sink)",
        f"{settings.mongo_host}:{settings.mongo_port}/{settings.mongo_db_name}",
        "[green]ONLINE[/green]" if mongo_ok else "[red]OFFLINE[/red]",
    )
    table.add_row(
        "HPO Ontology (offline)",
        settings.hpo_obo_path,
        "[green]FILE FOUND[/green]" if __import__("pathlib").Path(settings.hpo_obo_path).exists()
        else "[yellow]NOT FOUND — run: python scripts/download_ontologies.py[/yellow]",
    )
    table.add_row(
        "MONDO Ontology (offline)",
        settings.mondo_obo_path,
        "[green]FILE FOUND[/green]" if __import__("pathlib").Path(settings.mondo_obo_path).exists()
        else "[yellow]NOT FOUND — run: python scripts/download_ontologies.py[/yellow]",
    )

    console.print(table)


@app.command()
def run_demo(limit: int = typer.Option(5, "--limit", "-l", help="Number of records to extract and process")):
    """Run the complete end-to-end medical science pipeline."""
    pipeline = MedicalAnalysisPipeline()
    pipeline.run(limit=limit)


@app.command()
def embed(term: str = typer.Argument(..., help="Clinical condition or entity term to embed")):
    """Generate and display SapBERT embedding for a medical phrase."""
    console.print(f"Embedding clinical phrase: [cyan]'{term}'[/cyan]")
    embedder = SapBERTEmbedder()
    vec = embedder.embed_entities([term])[0]
    console.print(f"✓ Generated vector dimension: [green]{len(vec)}[/green]")
    console.print(f"Sample (first 5 components): [yellow]{vec[:5].round(4).tolist()}[/yellow]")


@app.command()
def search_hpo(term: str = typer.Argument(..., help="Phenotype or symptom term")):
    """Search the Human Phenotype Ontology (HPO) for a given term."""
    from meldai.terminology.hpo import get_hpo_service

    settings = get_settings()
    console.print(f"Searching HPO for: [cyan]'{term}'[/cyan]")
    service = get_hpo_service(settings.hpo_obo_path)
    results = service.search(term, limit=5)

    if not results:
        console.print("[red]No HPO concept found.[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"HPO Matches for '{term}'", show_lines=True)
    table.add_column("HPO ID", style="green")
    table.add_column("Label", style="white")
    table.add_column("Match Type", style="magenta")
    table.add_column("Synonyms (first 2)", style="dim")

    for r in results:
        table.add_row(r.hpo_id, r.label, r.match_type, ", ".join(r.synonyms[:2]))

    console.print(table)


@app.command()
def search_mondo(term: str = typer.Argument(..., help="Disease or disorder term")):
    """Search the MONDO Disease Ontology for a given term."""
    from meldai.terminology.mondo import get_mondo_service

    settings = get_settings()
    console.print(f"Searching MONDO for: [cyan]'{term}'[/cyan]")
    service = get_mondo_service(settings.mondo_obo_path)
    results = service.search(term, limit=5)

    if not results:
        console.print("[red]No MONDO concept found.[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"MONDO Matches for '{term}'", show_lines=True)
    table.add_column("MONDO ID", style="green")
    table.add_column("Label", style="white")
    table.add_column("Match Type", style="magenta")
    table.add_column("Synonyms (first 2)", style="dim")

    for r in results:
        table.add_row(r.mondo_id, r.label, r.match_type, ", ".join(r.synonyms[:2]))

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(True, "--reload"),
):
    """Start the FastAPI Web App."""
    console.print(f"🚀 Starting MeldAI Web Server on http://{host}:{port} ...")
    uvicorn.run("meldai.main:create_web_app", host=host, port=port, reload=reload, factory=True)


def cli_entrypoint():
    """Executable CLI entrypoint for pyproject.toml."""
    app()


if __name__ == "__main__":
    cli_entrypoint()
