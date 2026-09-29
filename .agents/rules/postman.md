# Postman Collection Synchronization Rule

## Objective
Ensure the Postman collection (`postman/meldai_postman_collection.json`) always stays synchronized with the live FastAPI endpoints in `src/meldai/api/`.

## Guidelines
1. **Trigger Condition**: Whenever any endpoint, route definition, request model (Pydantic schema), query parameter, or response model in `src/meldai/api/` or `src/meldai/main.py` is added, modified, or removed:
   - The Postman collection file `postman/meldai_postman_collection.json` MUST be updated.
   - Do not leave the Postman collection in an outdated state.

2. **Collection File Location**:
   - Primary collection file: `postman/meldai_postman_collection.json`

3. **Format & Quality Standards**:
   - Use the standard **Postman Collection v2.1.0** schema (`https://schema.getpostman.com/json/collection/v2.1.0/collection.json`).
   - Use parameterized variables for host URLs (`{{base_url}}`), defaulting to `http://localhost:8000`.
   - Include clear summaries, markdown descriptions, headers (`Content-Type: application/json`), realistic request bodies, query parameters, and sample HTTP 200/4xx response examples.
   - Group endpoints logically into folders if the API surface expands.

4. **Automation**:
   - You can update `postman/meldai_postman_collection.json` manually or by running `python scripts/export_postman.py` (or inside docker: `docker compose exec app python scripts/export_postman.py`).
