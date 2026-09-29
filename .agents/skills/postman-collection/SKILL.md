---
name: postman-collection
description: >-
  Maintains, regenerates, and validates the Postman API collection for the MeldAI platform.
  Use this skill whenever API endpoints, request schemas, or response models in src/meldai/api/
  are added, modified, or need to be exported into Postman JSON format.
---

# Postman Collection Management Skill

This skill guides the maintenance and export of the official Postman API collection for the MeldAI Clinical Intelligence platform.

## Key Files
- Collection Destination: `postman/meldai_postman_collection.json`
- Generator Script: `scripts/export_postman.py`
- FastAPI Router: `src/meldai/api/router.py`

## Workflow

### 1. Generating or Updating the Collection
When changes are made to API endpoints:
- Run the export script inside the running container:
  ```bash
  docker compose exec app python scripts/export_postman.py
  ```
- Or run it via make:
  ```bash
  make export-postman
  ```

### 2. Manual Verification
Always verify that:
1. Every endpoint in `src/meldai/api/router.py` exists in `postman/meldai_postman_collection.json`.
2. URLs use the `{{base_url}}` variable (e.g. `{{base_url}}/api/v1/embed`).
3. Request bodies reflect active Pydantic request models (`EmbedRequest`).
4. Example responses are included for immediate testing without running backend services.
5. The JSON conforms to Postman Collection Schema v2.1.0.
