#!/usr/bin/env python3
"""Download HPO and MONDO OBO files to data/ontologies/.

Usage:
    python scripts/download_ontologies.py

Files saved:
    data/ontologies/hp.obo     — Human Phenotype Ontology
    data/ontologies/mondo.obo  — MONDO Disease Ontology (includes MFO-MD mental disease terms)
"""

import sys
import urllib.request
from pathlib import Path

ONTOLOGIES = {
    "hp.obo": "https://purl.obolibrary.org/obo/hp.obo",
    "mondo.obo": "https://purl.obolibrary.org/obo/mondo.obo",
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ontologies"


def _progress(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        percent = min(100, int(count * block_size * 100 / total_size))
        downloaded_mb = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading... {percent}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
        sys.stdout.flush()


def download(name: str, url: str) -> None:
    dest = OUTPUT_DIR / name
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [skip] {name} already exists at {dest}")
        return

    print(f"  Fetching {name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        sys.stdout.write("\n")
        # Ensure file is readable/writable by host user as well
        dest.chmod(0o666)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name} saved ({size_mb:.1f} MB)")
    except Exception as exc:
        sys.stdout.write("\n")
        print(f"  ✗ Failed to download {name}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        OUTPUT_DIR.chmod(0o777)
        OUTPUT_DIR.parent.chmod(0o777)
    except Exception:
        pass
    print(f"Saving ontology files to: {OUTPUT_DIR}\n")
    for name, url in ONTOLOGIES.items():
        download(name, url)
    print("\nDone. You can now start the server:")
    print("  meldai serve   or   docker compose up")


if __name__ == "__main__":
    main()
