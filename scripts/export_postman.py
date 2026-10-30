#!/usr/bin/env python3
"""Export MeldAI FastAPI OpenAPI specifications to a Postman Collection v2.1.0 JSON."""

import json
from pathlib import Path
from typing import Any, Dict, List

from main import create_web_app


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
        "Case Similarity & Search": [],
        "Prescription & Batch Pipelines": [],
        "Medicine Master & BioLORD": [],
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

    # 4. Unified Patient Migration (Synchronous)
    folders["Data Migration & Sync"].append(
        {
            "name": "Migrate Patients (PostgreSQL -> Decorate -> MongoDB)",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate?batch_size=20",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate"],
                    "query": [
                        {
                            "key": "batch_size",
                            "value": "20",
                            "description": "Batch size for extracting, decorating, and saving records (default: 20)",
                        }
                    ],
                },
                "description": "Execute unified patient migration logical flow: start -> query docs from psql -> decorate -> save to mongodb -> extended. Repeated in batches of 20.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate?batch_size=20",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "success",
                            "batch_size": 20,
                            "total_records_processed": 1386,
                            "batches_processed": 70,
                            "upserted_count": 1386,
                            "modified_count": 0,
                            "matched_count": 0,
                            "execution_time_seconds": 12.45,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 5. Unified Patient Migration (Background Runner)
    folders["Data Migration & Sync"].append(
        {
            "name": "Start Migrate Patients (Background)",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate/start?batch_size=20",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate", "start"],
                    "query": [
                        {
                            "key": "batch_size",
                            "value": "20",
                            "description": "Batch size for background patient migration (default: 20)",
                        }
                    ],
                },
                "description": "Trigger the asynchronous background migration worker for patient records in batches of 20.",
            },
            "response": [
                {
                    "name": "200 Started",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate/start?batch_size=20",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate", "start"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "started",
                            "message": "Patient migration started in background.",
                            "batch_size": 20,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    folders["Data Migration & Sync"].append(
        {
            "name": "Stop Migrate Patients",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate/stop",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate", "stop"],
                },
                "description": "Send graceful cancellation signal to halt patient background migration.",
            },
            "response": [
                {
                    "name": "200 Stop Requested",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate/stop",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate", "stop"],
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
            "name": "Get Migrate Patients Status",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json", "type": "text"}],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/migrate/status",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "migrate", "status"],
                },
                "description": "Poll the active run status, current step, and progress metrics of patient migration.",
            },
            "response": [
                {
                    "name": "200 Status",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/migrate/status",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "migrate", "status"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "running",
                            "batch_size": 20,
                            "current_batch": 5,
                            "total_batches": 70,
                            "total_records_processed": 100,
                            "documents_saved": 100,
                            "current_step": "extended",
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

    # 13. Rank Medications by Similarity
    folders["Case Similarity & Search"].append(
        {
            "name": "Rank Medications by Similarity",
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
                            "type": "chiefComplaint",
                            "query": "insomnia",
                            "k": 5,
                            "patient_age": 45.0,
                            "patient_gender": "Male",
                            "weights": {
                                "w_lca": 0.5,
                                "w_vector": 0.5,
                                "min_similarity": 0.5,
                            },
                            "top_cases_limit": 20,
                            "candidate_pool_limit": 500,
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/rank-meds",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "rank-meds"],
                },
                "description": "Rank top-k molecules by prescription score from similar cases matching LCA + vector (meeting min_similarity cutoff) and sorted post-match by patient demographics.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/rank-meds",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "rank-meds"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "type": "chiefComplaint",
                            "query": "insomnia",
                            "k": 5,
                            "curie": "HP:0002360",
                            "matched_term": "Sleep disturbance",
                            "similar_cases_count": 20,
                            "weights_applied": {
                                "w_lca": 0.5,
                                "w_vector": 0.5,
                                "min_similarity": 0.5,
                                "patient_info_used": 1.0,
                            },
                            "molecules": [
                                {
                                    "canonical_molecule": "Zolpidem",
                                    "is_fdc": False,
                                    "score": 12.32,
                                    "frequency": 14,
                                    "percentage": 70.0,
                                    "avg_similarity": 0.88,
                                    "doctor_brands": ["ZOLFRESH", "ZOLRIN"],
                                },
                                {
                                    "canonical_molecule": "Etizolam",
                                    "is_fdc": False,
                                    "score": 7.38,
                                    "frequency": 9,
                                    "percentage": 45.0,
                                    "avg_similarity": 0.82,
                                    "doctor_brands": ["ZOLAVIL", "ETICALM"],
                                },
                            ],
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # -----------------------------------------------------------------------
    # Prescription & Batch Pipelines (V1 & V2)
    # -----------------------------------------------------------------------

    # 10. [V1] Prescribe Medications using Gemini LLM
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V1] Prescribe Medications (PostgreSQL + Gemini LLM)",
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
                            "case": {
                                "caseNo": 101,
                                "caseDate": "2024-03-15",
                                "visitType": "follow-up",
                                "patientInfo": {
                                    "age": 45.0,
                                    "gender": "Male",
                                },
                                "symptoms": ["lack of sleep", "fatigue"],
                                "diagnosis": ["Insomnia"],
                            },
                            "top_k": 5,
                            "params": {
                                "wDiagnosisScore": 0.4,
                                "wSymptomsScore": 0.4,
                                "wPatientScore": 0.1,
                                "wVisitTypeScore": 0.1,
                                "sameVisitTypeOnly": False,
                            },
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-medications",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-medications"],
                },
                "description": (
                    "Decorate target clinical case, retrieve patient past encounters and similar cases with prior medications, "
                    "construct context-rich prompt without embeddings, and prescribe medications using Gemini LLM."
                ),
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-medications",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-medications"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "medications": ["Melatonin 3mg", "Zolpidem 5mg"],
                            "explanation": "Medications selected for adult patient presenting with insomnia and fatigue, matching doctor's historical prescription pattern in similar encounters.",
                            "pastCasesCount": 1,
                            "similarCasesCount": 3,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 11. [V1] Start Prescribe Pipeline
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V1] Start Prescribe Pipeline (Background Batch)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/start",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "start"],
                },
                "description": "Start the V1 background pipeline that runs legacy SQL-driven prescribe_medications for cases in prescribe-request-all-cases.json.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/start",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "start"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "started",
                            "message": "Prescribe pipeline started in background.",
                            "run_id": "9421de3fac10",
                            "result_file": "/app/tmp/prescribe_pipeline_20261026_120000.json",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 12. [V1] Stop Prescribe Pipeline
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V1] Stop Prescribe Pipeline",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/stop",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "stop"],
                },
                "description": "Signal the active V1 prescribe pipeline to stop after the current case finishes.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/stop",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "stop"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "stopping",
                            "message": "Stop signal sent. Pipeline will halt after the current case. Results written so far are preserved.",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 13. [V1] Get Prescribe Pipeline Status
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V1] Get Prescribe Pipeline Status",
            "request": {
                "method": "GET",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/status",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "status"],
                },
                "description": "Poll the state, counters, current case, and output path for the V1 pipeline.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/status",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "status"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "running",
                            "run_id": "9421de3fac10",
                            "total_cases": 95,
                            "processed_cases": 32,
                            "passed_cases": 32,
                            "failed_cases": 0,
                            "current_case_no": 101,
                            "result_file": "/app/tmp/prescribe_pipeline_20261026_120000.json",
                            "elapsed_seconds": 120.0,
                            "error": None,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 14. [V2] Prescribe Medications (Multi-Step Reasoning)
    v2_prescribe_body = {
        "case": {
            "caseNo": 332,
            "caseDate": "2026-01-24",
            "visitType": "new consultation",
            "patientInfo": {
                "age": 37.0,
                "gender": "Male",
            },
            "symptoms": [
                "Sadness of mood, anxiety. restlessness. Total duration: 1.5 years, increased since 1.5 months",
                "Severe insomnia",
                "Occasional death wishes",
            ],
            "diagnosis": [
                "MDD WITH ANXIOUS DISTRESS",
                "INSOMNIA",
            ],
        },
        "top_k_per_entity": 5,
        "enable_deprescribing": True,
    }

    v2_prescribe_response = {
        "clinical_summary": "37yo Male presenting for new consultation with MDD WITH ANXIOUS DISTRESS and INSOMNIA. Reports 1.5 years of depressive symptoms, severe insomnia, and occasional death wishes.",
        "llmMedCheck": {
            "prompt": "ROLE: Clinical psychopharmacology assistant reviewing candidate medications...",
            "output": {
                "entity_medication_check": [
                    {
                        "entity": "Severe insomnia",
                        "attributed_molecules": ["Clonazepam", "Quetiapine"],
                        "recommended_top_molecules": [
                            {
                                "canonical_molecule": "Zolpidem",
                                "is_fdc": False,
                                "doctor_brands": ["ZOLAST"],
                                "score": 0.88,
                            }
                        ],
                    }
                ]
            },
        },
        "llmMedPrescribe": {
            "prompt": "ROLE: Assistant system for a psychiatrist practicing in India...",
            "output": {
                "medications": [
                    {"brand_name": "LERITHIUM", "canonical_molecule": "Lithium Carbonate"},
                    {"brand_name": "REVEN XR", "canonical_molecule": "Venlafaxine"},
                ],
                "explanation": "Standard therapeutic anchor for treatment-resistant major depression with anxious distress.",
            },
        },
        "medications": [
            {
                "brand_name": "LERITHIUM",
                "canonical_molecule": "Lithium Carbonate",
                "is_fdc": False,
                "constituent_strengths_mg": {"Lithium Carbonate": 400.0},
                "modal_strength_mg": 400.0,
                "release_form": "SR",
                "dosage_tier": "STANDARD_MAINTENANCE",
                "dosage_group_id": "GRP_LITHIUM_SR",
                "clinical_dosing_indication": "Sustained-release maintenance mood stabilizer for affective disorder.",
            },
            {
                "brand_name": "REVEN XR",
                "canonical_molecule": "Venlafaxine",
                "is_fdc": False,
                "constituent_strengths_mg": {"Venlafaxine": 75.0},
                "modal_strength_mg": 75.0,
                "release_form": "ER",
                "dosage_tier": "STANDARD_MAINTENANCE",
                "dosage_group_id": "GRP_VENLAFAXINE_ER_75",
                "clinical_dosing_indication": "Extended-release SNRI formulation for major depression and generalized anxiety.",
            },
        ],
        "entity_matrix": [
            {
                "entity": "Severe insomnia",
                "entity_type": "symptom",
                "curie": "HP:0002360",
                "delta_status": "NEW",
                "top_molecules": [
                    {
                        "canonical_molecule": "Clonazepam",
                        "is_fdc": False,
                        "frequency": 24,
                        "percentage": 48.0,
                        "score": 0.85,
                        "doctor_brands": ["CECON", "CLONATOP"],
                    }
                ],
                "attributed_molecules": ["Clonazepam"],
            }
        ],
        "explanation": "Patient presents with persistent severe depression and anxiety with insomnia. Venlafaxine ER anchor is prescribed alongside Lithium Carbonate mood stabilizer.",
        "past_cases_count": 0,
        "run_id": "case_332_20261026_120000_abc123",
        "metrics": {
            "response_time_seconds": 15.34,
            "tokens_used": {
                "prompt_tokens": 14200,
                "candidates_tokens": 820,
                "total_tokens": 15020,
            },
            "cost_usd": 0.00341,
            "two_step_execution": True,
        },
        "workflow_prompts_file": "/app/tmp/case_332_20261026_120000_abc123_all_prompts.txt",
        "llm_log_file": "/app/tmp/case_332_20261026_120000_abc123_llm_logs.json",
    }

    # 13b. [V2] Step 1: Triage & Deprescribe
    v2_deprescribe_body = {
        "case": {
            "caseNo": 332,
            "caseDate": "2026-01-24",
            "visitType": "follow-up",
            "patientInfo": {
                "age": 37.0,
                "gender": "Male",
            },
            "symptoms": [
                "Sadness of mood improved, anxiety reduced",
                "Mild insomnia remaining",
            ],
            "diagnosis": [
                "MDD WITH ANXIOUS DISTRESS",
                "INSOMNIA",
            ],
        },
    }

    v2_deprescribe_response = {
        "whats_changed": "Evaluated 2 prior medications: Clonazepam 0.5mg recommended for tapering due to improved anxiety; Escitalopram 10mg maintained for MDD stability.",
        "deprescribing_recommendations": [
            {
                "medicine": "Clonazepam 0.5mg",
                "target_symptom": "Anxiety",
                "status_change": "decreased",
                "action": "taper",
                "rationale": "Anxiety is well managed; step down benzodiazepine to avoid tolerance.",
                "taper_plan": "Reduce to 0.25mg for 7 days then stop.",
            },
            {
                "medicine": "Escitalopram 10mg",
                "target_symptom": "Depression",
                "status_change": "maintained",
                "action": "maintain",
                "rationale": "Patient shows clinical response; continue therapeutic dose.",
                "taper_plan": None,
            },
        ],
        "triage_delta": [
            {
                "complaint": "Sadness of mood improved, anxiety reduced",
                "category": "symptom",
                "canonical_entity": "Sadness of mood",
                "heuristic_mark": "improved",
                "llm_status": "improved",
                "what_changed": "Patient reports noticeable elevation in mood and marked anxiety reduction.",
            },
            {
                "complaint": "Mild insomnia remaining",
                "category": "symptom",
                "canonical_entity": "Insomnia",
                "heuristic_mark": "maintained",
                "llm_status": "maintained",
                "what_changed": "Residual sleep maintenance difficulty persists.",
            },
        ],
        "llm_response": {
            "prompt": "ROLE: Clinical psychopharmacology assistant reviewing prior encounters...",
            "output": {
                "whats_changed": "Evaluated 2 prior medications: Clonazepam 0.5mg recommended for tapering; Escitalopram 10mg maintained.",
                "deprescribing_recommendations": [
                    {
                        "medicine": "Clonazepam 0.5mg",
                        "target_symptom": "Anxiety",
                        "status_change": "decreased",
                        "action": "taper",
                        "rationale": "Anxiety is well managed; step down benzodiazepine to avoid tolerance.",
                    }
                ],
            },
        },
        "state": {
            "case": v2_deprescribe_body["case"],
            "whats_changed": "Evaluated 2 prior medications: Clonazepam 0.5mg recommended for tapering; Escitalopram 10mg maintained.",
            "triage_delta": [
                {
                    "complaint": "Sadness of mood improved, anxiety reduced",
                    "category": "symptom",
                    "canonical_entity": "Sadness of mood",
                    "llm_status": "improved",
                }
            ],
            "deprescribing_recommendations": [
                {
                    "medicine": "Clonazepam 0.5mg",
                    "target_symptom": "Anxiety",
                    "status_change": "decreased",
                    "action": "taper",
                }
            ],
            "historical_prescription": ["Clonazepam 0.5mg", "Escitalopram 10mg"],
            "latest_documented_prescription": ["Escitalopram 10mg"],
            "patient_reported_current_use": ["Escitalopram 10mg"],
            "actual_discontinuation": ["Clonazepam 0.5mg"],
        },
        "historical_prescription": ["Clonazepam 0.5mg", "Escitalopram 10mg"],
        "latest_documented_prescription": ["Escitalopram 10mg"],
        "patient_reported_current_use": ["Escitalopram 10mg"],
        "actual_discontinuation": ["Clonazepam 0.5mg"],
    }

    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Step 1: Triage & Deprescribe",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(v2_deprescribe_body, indent=2),
                },
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/deprescribe",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "deprescribe"],
                },
                "description": (
                    "V2 Step 1: Unified Triage & Deprescribing evaluation. Evaluates current intake complaints against "
                    "prior encounters to detect symptom changes (triage_delta: active, resolved, worsening, persistent, improving, new) "
                    "and determine medication actions (stop, taper, maintain, increase, new) in a single unified prompt. "
                    "Returns what's changed, triage_delta, deprescribing_recommendations, and the updated memory state."
                ),
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/deprescribe",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "deprescribe"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(v2_deprescribe_response, indent=2),
                }
            ],
        }
    )

    # 13c. [V2] Step 2: Med-Check & Missing Medication Filling
    v2_med_check_body = {
        "case": v2_prescribe_body["case"],
        "top_k_per_entity": 5,
        "enable_deprescribing": True,
    }

    v2_med_check_response = {
        "whats_changed": "Entity matrix constructed with 3 entities. Entity 'Severe insomnia' had 0 candidate medications in similarity search; batch filled 3 candidate molecules from local_medicine_master.json (Zolpidem, Etizolam, Melatonin).",
        "filled_entities": [
            {
                "entity": "Severe insomnia",
                "entity_type": "symptom",
                "curie": "HP:0002360",
                "missing_molecules_filled": ["Zolpidem", "Etizolam", "Melatonin"],
                "rationale": "Filled sedative/hypnotic candidates from local formulary master.",
            }
        ],
        "entity_matrix": [
            {
                "entity": "Severe insomnia",
                "entity_type": "symptom",
                "curie": "HP:0002360",
                "delta_status": "NEW",
                "top_molecules": [
                    {
                        "canonical_molecule": "Zolpidem",
                        "is_fdc": False,
                        "score": 0.88,
                        "doctor_brands": ["ZOLAST", "ZOLFRESH"],
                    }
                ],
                "attributed_molecules": ["Zolpidem"],
            }
        ],
        "llm_response": {
            "prompt": "ROLE: Clinical psychopharmacology assistant reviewing candidate medications...",
            "output": {
                "whats_changed": "Batch filled missing candidates for Severe insomnia.",
                "entity_medication_check": [
                    {
                        "entity": "Severe insomnia",
                        "recommended_top_molecules": [
                            {
                                "canonical_molecule": "Zolpidem",
                                "doctor_brands": ["ZOLAST"],
                            }
                        ],
                    }
                ],
            },
        },
        "memory_state": {
            "case": v2_prescribe_body["case"],
            "whats_changed": "Batch filled missing candidates for Severe insomnia.",
            "whats_changed_med_check": "Batch filled missing candidates for Severe insomnia.",
            "entity_matrix": [
                {
                    "entity": "Severe insomnia",
                    "entity_type": "symptom",
                    "top_molecules": [{"canonical_molecule": "Zolpidem"}],
                }
            ],
        },
    }

    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Step 2: Med-Check (Batch Fill Missing Medications)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(v2_med_check_body, indent=2),
                },
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/med-check",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "med-check"],
                },
                "description": (
                    "V2 Step 2: Runs Step 1 (triage & deprescribing) internally if needed, finds entities with 0 medications in the entity matrix, "
                    "asks the LLM in batch using local_medicine_master.json to fill missing candidate medications, reports what changed as a string, "
                    "and returns the updated entity matrix and memory state."
                ),
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/med-check",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "med-check"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(v2_med_check_response, indent=2),
                }
            ],
        }
    )

    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Prescribe Medications (Multi-Step Reasoning)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(v2_prescribe_body, indent=2),
                },
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/prescribe-medications",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "prescribe-medications"],
                },
                "description": (
                    "V2 Multi-Step Medication Reasoning: performs patient history delta, chief complaints triage, "
                    "case decoration via HPO/MONDO/SapBERT, per-entity candidate ranking, Step 1 LLM med check, "
                    "Step 2 LLM prescribing synthesis, and medicine master enrichment."
                ),
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/prescribe-medications",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "prescribe-medications"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(v2_prescribe_response, indent=2),
                }
            ],
        }
    )

    # 14b. [V2] Prescribe Medications (V1 Route Alias)
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Prescribe Medications (V1 Route Alias)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json", "type": "text"},
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "body": {
                    "mode": "raw",
                    "raw": json.dumps(v2_prescribe_body, indent=2),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-medications/v2",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-medications", "v2"],
                },
                "description": "Direct alias under /api/v1 for V2 multi-step prescribe medications.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-medications/v2",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-medications", "v2"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(v2_prescribe_response, indent=2),
                }
            ],
        }
    )

    # 15. [V2] Start Prescribe Pipeline
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Start Prescribe Pipeline (95-Case Multi-Step Reasoning)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/start",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "prescribe-pipeline", "start"],
                },
                "description": (
                    "Start the V2 background batch pipeline running multi-step reasoning across all 95 unique cases "
                    "in prescribe-request-all-cases.json. Writes incremental results with prompts, req, and res into tmp/."
                ),
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/start",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "prescribe-pipeline", "start"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "started",
                            "message": "Prescribe pipeline V2 started in background.",
                            "run_id": "8348dc1cbcc9",
                            "result_file": "/app/tmp/prescribe_pipeline_v2_20261026_120000.json",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 16. [V2] Stop Prescribe Pipeline
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Stop Prescribe Pipeline",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/stop",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "prescribe-pipeline", "stop"],
                },
                "description": "Signal the active V2 prescribe pipeline to gracefully halt after the current case finishes.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/stop",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "prescribe-pipeline", "stop"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "stopping",
                            "message": "Stop signal sent. Pipeline V2 will halt after the current case. Results written so far are preserved.",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 17. [V2] Get Prescribe Pipeline Status
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Get Prescribe Pipeline Status",
            "request": {
                "method": "GET",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/status",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v2", "cases", "prescribe-pipeline", "status"],
                },
                "description": "Poll the progress counters, current case number, elapsed time, and result file path for the active V2 pipeline.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v2/cases/prescribe-pipeline/status",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v2", "cases", "prescribe-pipeline", "status"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "running",
                            "run_id": "8348dc1cbcc9",
                            "version": "v2",
                            "total_cases": 95,
                            "processed_cases": 15,
                            "passed_cases": 15,
                            "failed_cases": 0,
                            "current_case_no": 332,
                            "result_file": "/app/tmp/prescribe_pipeline_v2_20261026_120000.json",
                            "elapsed_seconds": 210.5,
                            "error": None,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 18. [V2] Start Prescribe Pipeline (V1 Route Alias)
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Start Prescribe Pipeline (V1 Route Alias)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/start",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "start"],
                },
                "description": "Start the V2 batch pipeline using the /api/v1 URL namespace.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/start",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "start"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "started",
                            "message": "Prescribe pipeline V2 started in background.",
                            "run_id": "8348dc1cbcc9",
                            "result_file": "/app/tmp/prescribe_pipeline_v2_20261026_120000.json",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 19. [V2] Stop Prescribe Pipeline (V1 Route Alias)
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Stop Prescribe Pipeline (V1 Route Alias)",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/stop",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "stop"],
                },
                "description": "Stop the V2 batch pipeline using the /api/v1 URL namespace.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/stop",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "stop"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "stopping",
                            "message": "Stop signal sent. Pipeline V2 will halt after the current case. Results written so far are preserved.",
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 20. [V2] Get Prescribe Pipeline Status (V1 Route Alias)
    folders["Prescription & Batch Pipelines"].append(
        {
            "name": "[V2] Get Prescribe Pipeline Status (V1 Route Alias)",
            "request": {
                "method": "GET",
                "header": [
                    {"key": "Accept", "value": "application/json", "type": "text"},
                ],
                "url": {
                    "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/status",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "status"],
                },
                "description": "Poll the V2 batch pipeline progress using the /api/v1 URL namespace.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "GET",
                        "url": {
                            "raw": "{{base_url}}/api/v1/cases/prescribe-pipeline/v2/status",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "cases", "prescribe-pipeline", "v2", "status"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": json.dumps(
                        {
                            "status": "running",
                            "run_id": "8348dc1cbcc9",
                            "version": "v2",
                            "total_cases": 95,
                            "processed_cases": 15,
                            "passed_cases": 15,
                            "failed_cases": 0,
                            "current_case_no": 332,
                            "result_file": "/app/tmp/prescribe_pipeline_v2_20261026_120000.json",
                            "elapsed_seconds": 210.5,
                            "error": None,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )


    # 15. Load Medicine Master
    folders["Medicine Master & BioLORD"].append(
        {
            "name": "Load Medicine Master (JSON to MongoDB)",
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
                            "file_path": "data/medication/local_medicine_master.json",
                            "batch_size": 100,
                            "recreate": False,
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/medicines/load-master",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "medicines", "load-master"],
                },
                "description": "Ingest local_medicine_master.json, generate BioLORD embeddings for synthesized clinical sentences, and store in MongoDB in batches of 100.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({"batch_size": 100, "recreate": False}),
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/medicines/load-master",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "medicines", "load-master"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "body": json.dumps(
                        {
                            "status": "success",
                            "collection": "medicine_master",
                            "total_records": 349,
                            "upserted_count": 349,
                            "modified_count": 0,
                            "total_in_db": 349,
                            "batches_processed": 4,
                            "duration_seconds": 3.42,
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 16. BioLORD Text Embeddings
    folders["Medicine Master & BioLORD"].append(
        {
            "name": "BioLORD Clinical Text Embeddings",
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
                            "texts": [
                                "Etizolam, marketed as ZOLAVIL, is used as short-term bridge therapy for generalized anxiety and associated insomnia.",
                                "Escitalopram, marketed as ESCIREN, is used as first-line maintenance therapy for major depressive disorder.",
                            ],
                            "normalize": True,
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/biolord/embed",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "biolord", "embed"],
                },
                "description": "Generate 768-d BioLORD dense vector representations using multi-core inference and mean pooling.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps(
                                {
                                    "texts": [
                                        "Etizolam, marketed as ZOLAVIL, is used as short-term bridge therapy for generalized anxiety and associated insomnia."
                                    ]
                                }
                            ),
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/biolord/embed",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "biolord", "embed"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "body": json.dumps(
                        {
                            "texts": [
                                "Etizolam, marketed as ZOLAVIL, is used as short-term bridge therapy for generalized anxiety and associated insomnia."
                            ],
                            "dimension": 768,
                            "embeddings": [[0.012, -0.045, 0.089]],
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 17. Search Medicine Master
    folders["Medicine Master & BioLORD"].append(
        {
            "name": "Search Medicine Master (Semantic Similarity)",
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
                            "query": "anxiety and insomnia bridge therapy",
                            "top_k": 5,
                            "min_score": 0.4,
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/medicines/search",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "medicines", "search"],
                },
                "description": "Search medicine master records by clinical symptom, molecule, or indication using BioLORD semantic cosine similarity.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps(
                                {
                                    "query": "anxiety and insomnia bridge therapy",
                                    "top_k": 5,
                                    "min_score": 0.4,
                                }
                            ),
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/medicines/search",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "medicines", "search"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "body": json.dumps(
                        {
                            "query": "anxiety and insomnia bridge therapy",
                            "total_matches": 1,
                            "matches": [
                                {
                                    "brand_name": "ZOLAVIL",
                                    "canonical_molecule": "Etizolam",
                                    "is_fdc": False,
                                    "constituent_strengths_mg": {"Etizolam": 0.5},
                                    "modal_strength_mg": 0.5,
                                    "release_form": "Regular",
                                    "dosage_tier": "STANDARD_MAINTENANCE",
                                    "dosage_group_id": "GRP_ETIZOLAM_STANDARD",
                                    "clinical_dosing_indication": "Short-term bridge therapy for generalized anxiety and associated insomnia.",
                                    "digest_text": "Etizolam, marketed as ZOLAVIL, is used as short-term bridge therapy for generalized anxiety and associated insomnia.",
                                    "similarity_score": 0.8412,
                                }
                            ],
                        },
                        indent=2,
                    ),
                }
            ],
        }
    )

    # 18. Generate Medication Search Queries
    folders["Medicine Master & BioLORD"].append(
        {
            "name": "Generate Medication Search Queries (LLM Vector & BM25)",
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
                            "symptoms": ["anxiety", "restlessness", "insomnia"],
                            "diagnosis": ["MDD WITH ANXIOUS DISTRESS"],
                            "durationContext": "1.5 years",
                            "currentRegimen": ["Venlafaxine 75mg"],
                        },
                        indent=2,
                    ),
                },
                "url": {
                    "raw": "{{base_url}}/api/v1/medicines/generate-queries",
                    "host": ["{{base_url}}"],
                    "path": ["api", "v1", "medicines", "generate-queries"],
                },
                "description": "Execute active clinical prompt with Gemini LLM to synthesize vector semantic search queries and BM25 keywords for medicine matching.",
            },
            "response": [
                {
                    "name": "200 OK",
                    "originalRequest": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps(
                                {
                                    "symptoms": ["anxiety", "restlessness", "insomnia"],
                                    "diagnosis": ["MDD WITH ANXIOUS DISTRESS"],
                                }
                            ),
                        },
                        "url": {
                            "raw": "{{base_url}}/api/v1/medicines/generate-queries",
                            "host": ["{{base_url}}"],
                            "path": ["api", "v1", "medicines", "generate-queries"],
                        },
                    },
                    "status": "OK",
                    "code": 200,
                    "_postman_previewlanguage": "json",
                    "body": json.dumps(
                        {
                            "vector_semantic_queries": [
                                "First-line therapeutic maintenance anchor for major depressive episodes and generalized anxiety disorder",
                                "Intermediate- to long-acting benzodiazepine standard nocturnal dose for panic disorder and generalized anxiety bridging",
                            ],
                            "bm25_keywords": [
                                "Escitalopram",
                                "Clonazepam",
                                "Venlafaxine",
                                "SSRI",
                                "SNRI",
                                "Benzodiazepine",
                            ],
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
