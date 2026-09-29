#!/usr/bin/env python3
"""Export MeldAI FastAPI OpenAPI specifications to a Postman Collection v2.1.0 JSON."""

import json
from pathlib import Path
from typing import Any, Dict, List

from meldai.main import create_web_app


def build_postman_collection() -> Dict[str, Any]:
    """Build Postman Collection v2.1.0 from FastAPI routes and metadata."""
    app = create_web_app()
    openapi = app.openapi()

    collection = {
        "info": {
            "_postman_id": "7f8c12a4-569d-4e9b-b8f2-39c8194ad901",
            "name": "MeldAI Clinical Intelligence Platform API",
            "description": openapi.get("info", {}).get("description", "MeldAI REST API"),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "version": openapi.get("info", {}).get("version", "0.1.0"),
        },
        "variable": [
            {
                "key": "base_url",
                "value": "http://localhost:8000",
                "type": "string",
                "description": "Base URL of the MeldAI FastAPI web service",
            }
        ],
        "item": [],
    }

    folders: Dict[str, List[Dict[str, Any]]] = {
        "System & Health": [],
        "Clinical NLP & Embeddings": [],
        "Terminology (SNOMED CT)": [],
        "Data Migration & Sync": [],
    }

    # 1. Health Check
    folders["System & Health"].append(
        {
            "name": "Health Check",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/health",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "health"],
                },
                "description": "Check connectivity to PostgreSQL, MongoDB, and Snowstorm.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/health",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "health"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "online",
                            "environment": "development",
                            "services": {"postgres": False, "mongodb": True, "snowstorm": True},
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 2. Embed
    folders["Clinical NLP & Embeddings"].append(
        {
            "name": "Generate SapBERT Embeddings",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(
                        {"terms": ["acute myocardial infarction", "chest pain", "substernal pressure"]},
                        indent=2,
                    ),
                    "options": {"raw": {"language": "json"}},
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/embed",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "embed"],
                },
                "description": "Generate dense SapBERT clinical embeddings for given medical phrases.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/embed",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "embed"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "terms": ["acute myocardial infarction", "chest pain"],
                            "dimension": 768,
                            "embeddings": [
                                [-0.0312, 0.0541, 0.0124, -0.0194],
                                [0.0170, 0.0357, -0.0203, -0.0452],
                            ],
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 3. SNOMED Search
    folders["Terminology (SNOMED CT)"].append(
        {
            "name": "Search SNOMED CT Concept",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/snomed/search?term=chest%20pain&limit=5",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "snomed", "search"],
                    "query": [
                        {
                            "key": "term",
                            "value": "chest pain",
                            "description": "Clinical diagnosis or symptom term (minimum 2 characters)",
                        },
                        {"key": "limit", "value": "5", "description": "Maximum concept matches to return"},
                    ],
                },
                "description": "Search SNOMED CT terminology via Snowstorm.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/snomed/search?term=chest%20pain&limit=5",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "snomed", "search"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        [
                            {
                                "concept_id": "29857009",
                                "fsn": "Chest pain (finding)",
                                "preferred_term": "Chest pain",
                                "semantic_tag": "finding",
                                "score": 1.0,
                                "definition_status": "PRIMITIVE",
                            }
                        ],
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 4. Patient Visit Migration
    folders["Data Migration & Sync"].append(
        {
            "name": "Migrate Patient Visits (PostgreSQL -> MongoDB)",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate-patient-visits?batch_size=1000",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate-patient-visits"],
                    "query": [
                        {
                            "key": "batch_size",
                            "value": "1000",
                            "description": "Batch size for extracting and pushing records",
                        }
                    ],
                },
                "description": "Pull patient visit data from PostgreSQL (temp_migrations.patient_visit_data) in batches of 1000 and upsert into MongoDB 'cases' collection with unique (caseNo, visitDate) index.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate-patient-visits?batch_size=1000",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate-patient-visits"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "success",
                            "batch_size": 1000,
                            "total_records_processed": 1386,
                            "batches_processed": 2,
                            "upserted_count": 1386,
                            "modified_count": 0,
                            "matched_count": 0,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    for folder_name, items in folders.items():
        collection["item"].append(
            {
                "name": folder_name,
                "item": items,
            }
        )

    return collection


def main():
    """Generate and write the collection to postman/meldai_postman_collection.json."""
    collection = build_postman_collection()
    output_path = Path("postman/meldai_postman_collection.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    print(f"✓ Postman Collection v2.1.0 exported to: {output_path}")


if __name__ == "__main__":
    main()
