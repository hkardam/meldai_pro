"""MeldAI CLI and Application Entrypoint."""

import logging
from pathlib import Path
import sys
from typing import Optional

from fastapi import FastAPI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
import uvicorn

from core.config import get_settings
from core.db.mongodb import MongoKnowledgeBase
from core.db.postgres import PostgresSource
from core.nlp.sapbert import SapBERTEmbedder
from core.pipelines.medical_analysis import MedicalAnalysisPipeline
from core.router import core_router
from meldai_v1.router import v1_router
from meldai_v2.router import v2_router
from migrations.router import migration_router

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
    help="MeldAI: Clinical Intelligence — Core Services, MeldAI V1, and MeldAI V2",
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
        description="REST API with Core services, MeldAI V1 prescription, and MeldAI V2 multi-step reasoning",
        version="0.2.0",
    )

    # Core routes (mounted under /api/v1 for test/client backward compatibility and /api)
    web_app.include_router(core_router, prefix="/api/v1", tags=["core"])
    web_app.include_router(core_router, prefix="/api", tags=["core"], include_in_schema=False)

    # Migration routes
    web_app.include_router(migration_router, prefix="/api/v1", tags=["migrations"])
    web_app.include_router(migration_router, prefix="/api", tags=["migrations"], include_in_schema=False)

    # MeldAI V1 routes
    web_app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
    web_app.include_router(v1_router, prefix="/api", tags=["v1"], include_in_schema=False)

    # MeldAI V2 routes
    web_app.include_router(v2_router, prefix="/api/v2", tags=["v2"])

    return web_app


@app.command()
def check_health():
    """Verify connectivity to PostgreSQL and MongoDB, and show ontology config."""
    console.print(Panel.fit("[bold blue]🩺 MeldAI Infrastructure Health Check[/bold blue]"))
    st = get_settings()

    pg = PostgresSource(st)
    mongo = MongoKnowledgeBase(st)

    pg_ok = pg.ping()
    mongo_ok = mongo.ping()

    table = Table(title="Service Connectivity Status", show_lines=True)
    table.add_column("Component", style="cyan")
    table.add_column("Target", style="white")
    table.add_column("Status", style="bold")

    table.add_row(
        "PostgreSQL (External Source)",
        f"{st.postgres_host}:{st.postgres_port}/{st.postgres_db}",
        "[green]ONLINE[/green]" if pg_ok else "[yellow]OFFLINE (Configure .env for external PG)[/yellow]",
    )
    table.add_row(
        "MongoDB (Knowledge Base Sink)",
        f"{st.mongo_host}:{st.mongo_port}/{st.mongo_db_name}",
        "[green]ONLINE[/green]" if mongo_ok else "[red]OFFLINE[/red]",
    )
    table.add_row(
        "HPO Ontology (offline)",
        st.hpo_obo_path,
        "[green]FILE FOUND[/green]" if Path(st.hpo_obo_path).exists()
        else "[yellow]NOT FOUND — run: python scripts/download_ontologies.py[/yellow]",
    )
    table.add_row(
        "MONDO Ontology (offline)",
        st.mondo_obo_path,
        "[green]FILE FOUND[/green]" if Path(st.mondo_obo_path).exists()
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
    from core.terminology.hpo import get_hpo_service

    st = get_settings()
    console.print(f"Searching HPO for: [cyan]'{term}'[/cyan]")
    service = get_hpo_service(st.hpo_obo_path)
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
    from core.terminology.mondo import get_mondo_service

    st = get_settings()
    console.print(f"Searching MONDO for: [cyan]'{term}'[/cyan]")
    service = get_mondo_service(st.mondo_obo_path)
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
    src_dir = Path(__file__).resolve().parent
    uvicorn.run(
        "main:create_web_app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(src_dir)],
        reload_excludes=["*/tmp/*", "tmp/*", "tmp", "*tmp*"],
        factory=True,
    )


def cli_entrypoint():
    """Executable CLI entrypoint for pyproject.toml."""
    app()


if __name__ == "__main__":
    cli_entrypoint()
