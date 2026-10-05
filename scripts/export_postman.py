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
        "Terminology (HPO & MONDO)": [],
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
                "description": "Check connectivity to PostgreSQL, MongoDB, and ontology paths.",
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
                            "services": {"postgres": True, "mongodb": True},
                            "ontologies": {
                                "hpo": "data/hpo/hp.obo",
                                "mondo": "data/mondo/mondo.obo",
                            },
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

    # 2b. Segment Symptoms
    folders["Clinical NLP & Embeddings"].append(
        {
            "name": "Segment Symptoms",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(
                        {"note": "● anxiety while driving\n● 60% better mood"},
                        indent=2,
                    ),
                    "options": {"raw": {"language": "json"}},
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/utils/segment-symptoms",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "utils", "segment-symptoms"],
                },
                "description": "Split clinical note into segments, classify phenotypes, and attach HPO code + SapBERT embedding for phenotypes.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/utils/segment-symptoms",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "utils", "segment-symptoms"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "symptoms": [
                                {
                                    "note": "anxiety while driving",
                                    "isPheno": True,
                                    "embedding": [-0.0312, 0.0541, 0.0124],
                                    "hpoCode": 745,
                                },
                                {
                                    "note": "60% better mood",
                                    "isPheno": False,
                                },
                            ]
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    folders["Clinical NLP & Embeddings"].append(
        {
            "name": "Batch Match Symptoms (medspaCy Pipe + HPO + MONDO Top 3)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(
                        {
                            "symptoms": [
                                "severe headache",
                                "denies fever",
                                "chronic insomnia"
                            ],
                            "top_k": 3
                        },
                        indent=2,
                    ),
                    "options": {"raw": {"language": "json"}},
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/symptoms/match-batch",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "symptoms", "match-batch"],
                },
                "description": "Batch process an array of symptom strings through medspaCy assertion NLP pipe and return top 3 HPO & MONDO concept matches.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/symptoms/match-batch",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "symptoms", "match-batch"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "total_symptoms": 3,
                            "matches": [
                                {
                                    "symptom": "severe headache",
                                    "search_target": "headache",
                                    "assertion_status": "affirmed",
                                    "is_negated": False,
                                    "is_phenotype": True,
                                    "hpo_matches": [
                                        {
                                            "hpo_id": "HP:0002315",
                                            "code": 2315,
                                            "label": "Headache",
                                            "match_type": "exact_label",
                                            "score": 1.0
                                        }
                                    ],
                                    "mondo_matches": [
                                        {
                                            "mondo_id": "MONDO:0005555",
                                            "code": 5555,
                                            "label": "headache disorder",
                                            "match_type": "exact_label",
                                            "score": 1.0
                                        }
                                    ]
                                }
                            ]
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )


    # 3. Terminology (HPO & MONDO)
    folders["Terminology (HPO & MONDO)"].append(
        {
            "name": "Search HPO Concept (medspaCy Assertion-Aware)",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/hpo/search?term=playing%20video%20games&limit=5&filter_negated=false",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "hpo", "search"],
                    "query": [
                        {
                            "key": "term",
                            "value": "playing video games",
                            "description": "Phenotype / symptom term to look up (evaluated with medspaCy ConText for negation)",
                        },
                        {"key": "limit", "value": "5", "description": "Maximum concept matches to return"},
                        {
                            "key": "filter_negated",
                            "value": "false",
                            "description": "If true, exclude concepts detected as negated (returns empty array [] if negated)",
                        },
                    ],
                },
                "description": "Search Human Phenotype Ontology (HPO) concepts with medspaCy clinical assertion (negation detection).",
            },
            "response": [
                {
                    "name": "200 OK (Affirmed Phenotype)",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/hpo/search?term=playing%20video%20games&limit=5&filter_negated=true",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "hpo", "search"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        [
                            {
                                "hpo_id": "HP:5200336",
                                "label": "Addictive video game use",
                                "synonyms": [
                                    "Excessive video game playing",
                                    "Video game addiction"
                                ],
                                "definition": "The inability to regulate persistent gaming behavior is characterized by a heightened prioritization of gaming activities over regular daily tasks and responsibilities.",
                                "match_type": "token_label",
                                "score": 0.8325,
                                "is_negated": False,
                                "is_phenotype": True,
                                "assertion_status": "affirmed"
                            }
                        ],
                        indent=2,
                    ),
                }
            ],
        }
    )

    folders["Terminology (HPO & MONDO)"].append(
        {
            "name": "Search MONDO Disease Concept",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/mondo/search?term=diabetes%20mellitus&limit=5",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "mondo", "search"],
                    "query": [
                        {
                            "key": "term",
                            "value": "diabetes mellitus",
                            "description": "Disease / disorder term to look up",
                        },
                        {"key": "limit", "value": "5", "description": "Maximum concept matches to return"},
                    ],
                },
                "description": "Search the MONDO Disease Ontology for a given disease / disorder term.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/mondo/search?term=diabetes%20mellitus&limit=5",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "mondo", "search"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        [
                            {
                                "mondo_id": "MONDO:0005015",
                                "label": "diabetes mellitus",
                                "synonyms": [
                                    "diabetes"
                                ],
                                "definition": "A metabolic disease characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both.",
                                "match_type": "exact_label",
                                "score": 1.0
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

    # 5. Patient Diagnosis Migration
    folders["Data Migration & Sync"].append(
        {
            "name": "Migrate Patient Diagnoses (PostgreSQL -> MongoDB)",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate-patient-diagnoses?batch_size=1000",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate-patient-diagnoses"],
                    "query": [
                        {
                            "key": "batch_size",
                            "value": "1000",
                            "description": "Batch size for extracting and pushing diagnosis records",
                        }
                    ],
                },
                "description": "Pull grouped patient diagnosis data from PostgreSQL (temp_migrations.patient_diagnosis_data) using ARRAY_AGG(DISTINCT 'Diagnosis Name'), resolve MONDO codes and SapBERT embeddings, and replace the 'diagnosis' array on matching MongoDB 'cases' documents.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate-patient-diagnoses?batch_size=1000",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate-patient-diagnoses"],
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
                            "total_encounters_processed": 523,
                            "batches_processed": 1,
                            "modified_count": 523,
                            "matched_count": 523,
                            "unique_terms_indexed": 57,
                            "execution_time_seconds": 1.452,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 6. Patient Chief Complaints Migration (Background Runner)
    folders["Data Migration & Sync"].append(
        {
            "name": "Start Migrate Patient Chief Complaints (Background)",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/start?batch_size=100",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "start"],
                    "query": [
                        {
                            "key": "batch_size",
                            "value": "100",
                            "description": "Batch size for extracting and pushing records in the background",
                        }
                    ],
                },
                "description": "Trigger the asynchronous background migration worker for patient chief complaints.",
            },
            "response": [
                {
                    "name": "200 Started",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/start?batch_size=100",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "start"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "started",
                            "message": "Chief complaints migration started in background.",
                            "batch_size": 100,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    folders["Data Migration & Sync"].append(
        {
            "name": "Stop Migrate Patient Chief Complaints",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/stop",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "stop"],
                },
                "description": "Send graceful cancellation signal to halt chief complaints background migration.",
            },
            "response": [
                {
                    "name": "200 Stop Requested",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/stop",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "stop"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "stopping",
                            "message": "Stop signal sent. Runner will cleanly halt after finishing current batch.",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    folders["Data Migration & Sync"].append(
        {
            "name": "Get Migrate Patient Chief Complaints Status",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/status",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "status"],
                },
                "description": "Poll the active run status and progress metrics of the chief complaints migration.",
            },
            "response": [
                {
                    "name": "200 Status",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate-patient-chief-complaints/status",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate-patient-chief-complaints", "status"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "running",
                            "batch_size": 100,
                            "current_batch": 3,
                            "total_batches": 10,
                            "total_rows_processed": 300,
                            "documents_updated": 290,
                            "documents_skipped": 10,
                            "current_step": "batch_3_completed",
                            "start_time": 1728120000.0,
                            "elapsed_seconds": 3.8,
                            "error": None,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    for folder_name, items in folders.items():
        if items:
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
