# 🩺 MeldAI Medical Science Platform

A modern, human-friendly, dockerized Python environment built for **medical data science, clinical NLP, and algorithm development**.

Structured to effortlessly evolve into a high-performance web service (FastAPI) when you are ready.

---

## 🏛️ System Architecture

```mermaid
graph LR
    subgraph External Data Sources
        PG[(PostgreSQL\nExternal Training Source)]
    end

    subgraph Clinical Science Core (Docker)
        APP[MeldAI Pipeline]
        SAP[SapBERT\nPubMedBERT Embeddings]
        SNOMED[Snowstorm\nSNOMED CT REST]
    end

    subgraph Internal Knowledge Base (Docker)
        MONGO[(MongoDB 7.0\nCases Knowledge Base)]
    end

    PG -.->|External Connection| APP
    APP -->|Generate 768-d Vectors| SAP
    APP -->|Standardize Concepts| SNOMED
    APP -->|Sink Enriched Cases| MONGO
```

---

## 📄 MongoDB Case Document Structure

Each clinical case processed by the pipeline is standardized and stored in the internal MongoDB `cases` collection:

```json
{
  "id": "uuid",
  "caseNo": 101,
  "visitDate": "2026-09-25",
  "patient": {
    "name": "John Doe",
    "gender": "Male",
    "age": {
      "year": 58,
      "month": 6,
      "day": 14
    }
  },
  "symptoms": [
    {
      "text": "substernal chest pressure",
      "snomedTitle": "Chest pain",
      "snomedCode": 29857009,
      "embedding": [-0.0142, 0.0891, "...768-dim vector..."]
    }
  ],
  "diagnosis": [
    {
      "text": "acute myocardial infarction",
      "snomedTitle": "Myocardial infarction",
      "snomedCode": 22298006,
      "embedding": [0.0345, -0.0112, "...768-dim vector..."]
    }
  ]
}
```

---

## 🔌 Connecting to External PostgreSQL

PostgreSQL has been decoupled from Docker Compose to allow direct connection to your existing external database.

Configure your external PostgreSQL connection in `.env`:

```env
# If PostgreSQL is running on your host machine:
POSTGRES_HOST=host.docker.internal
POSTGRES_PORT=5432
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_pass
POSTGRES_DB=your_database_name
```

*(If PostgreSQL is on a remote server/cloud, simply set `POSTGRES_HOST=192.168.x.x` or `your-db.company.com`)*.

---

## 🚀 Quick Start (Dockerized)

### 1. Start Internal MongoDB & Application
```bash
make up
# or: docker compose up -d
```

### 2. Verify Health Status
Check that external PostgreSQL, internal MongoDB, and Snowstorm connectivity are operational:
```bash
docker compose run --rm app python -m meldai.main check-health
```

### 3. Run the Clinical Pipeline
Extract training cases, generate SapBERT embeddings, map to SNOMED CT, and sink into MongoDB:
```bash
make run-pipeline
# or: docker compose run --rm app python -m meldai.main run-demo
```

---

## 🐛 Debugging Inside Docker (VS Code & IDEs)

This project has first-class support for **debugging live Python code running inside Docker** using `debugpy`.

### How to Debug in VS Code:
1. Open the project in VS Code.
2. Put a **breakpoint** (red dot) anywhere in your code (e.g. inside `src/meldai/pipelines/medical_analysis.py`).
3. Run the container in debug wait mode:
   ```bash
   make debug
   ```
   *(The container will start and print `[WAIT] Waiting for IDE debugger to attach...`)*
4. In VS Code, press **`F5`** (or go to **Run and Debug** -> select **"Python: Attach to Docker Container (debugpy)"**).
5. The debugger attaches immediately, hits your breakpoint, and allows you to inspect variables, step into functions, and evaluate expressions!

> **Live Code Reloading**: The host directory is mounted directly into `/app`. When you edit code in your editor, changes take effect immediately without rebuilding Docker images.

---

## 💻 CLI Commands & Experiments

### 1. Test SapBERT Clinical Embeddings
```bash
docker compose run --rm app python -m meldai.main embed "acute myocardial infarction"
```

### 2. Map Clinical Terms to SNOMED CT
```bash
docker compose run --rm app python -m meldai.main map-snomed "Type 2 diabetes mellitus"
```

### 3. Interactive Shell
```bash
make bash
# or: docker compose run --rm --service-ports app bash
```

### 4. Run Automated Tests
```bash
make test
# or: docker compose run --rm app pytest -v
```

---

## 🌐 Transitioning to a Web App Later

When you are ready to expose this as a web application:
1. Start the API server:
   ```bash
   docker compose run --rm -p 8000:8000 app python -m meldai.main serve --host 0.0.0.0 --port 8000
   ```
2. Interactive Swagger / OpenAPI documentation is immediately available at:
   - **http://localhost:8000/docs**
   - Endpoints include `/api/v1/health`, `/api/v1/embed`, and `/api/v1/snomed/search`.

3. **Postman API Collection**:
   - Ready-to-import Postman collection: `postman/meldai_postman_collection.json`
   - Regenerate or sync anytime:
     ```bash
     make export-postman
     ```

