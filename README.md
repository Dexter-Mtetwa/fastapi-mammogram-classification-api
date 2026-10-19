# cancer-catch
# Breast Cancer Classification API  

A production‑ready **FastAPI** service that receives mammography images, runs them through a pre‑trained TensorFlow `.h5` CNN model, and returns a cancer prediction (malignant / benign / negative). The API is deliberately built **without an ORM** – all database interaction uses raw SQL via **psycopg** – to keep the data‑access layer transparent and to showcase core backend engineering concepts.

> **Why this project?**  
> Early‑stage medical‑imaging pipelines often start with a “model‑as‑a‑file” approach. This repository demonstrates how to expose such a model as a **secure, container‑ready, testable, and well‑instrumented** web service that can be dropped into a larger clinical‑workflow or research platform.

---  

## Table of Contents  

- [Overview](#overview)  
- [Key Design Decisions](#key-design-decisions)  
- [Architecture](#architecture)  
- [Request Flow](#request-flow)  
- [Technology Stack](#technology-stack)  
- [Project Structure](#project-structure)  
- [Database Design](#database-design)  
- [Authentication & Authorization](#authentication--authorization)  
- [API Endpoints](#api-endpoints)  
- [Health & Readiness Checks](#health--readiness-checks)  
- [Configuration & Environment Variables](#configuration--environment-variables)  
- [Docker & Docker‑Compose](#docker--docker-compose)  
- [Testing](#testing)  
- [Running Locally](#running-locally)  
- [Running with Docker Compose](#running-with-docker-compose)  
- [Future Improvements](#future-improvements)  
- [License](#license)  
- [Author](#author)  

---  

## Overview  

The service accepts **DICOM** or **PNG/JPEG** mammography scans (single‑image or multi‑view studies), stores the raw bytes for auditability, and forwards the image to a **TensorFlow `model.h5`** CNN that was trained on thousands of Kaggle mammography scans. The model outputs a categorical prediction:

| Class | Meaning |
|------|----------|
| `0` | No cancer (benign) |
| `1` | Malignant (cancerous) |
| `2` | Indeterminate / needs review |

Predictions are persisted together with the request metadata (user, timestamp, image hash) for downstream analytics and compliance.

---  

## Key Design Decisions  

| Decision | Rationale |
|----------|-----------|
| **FastAPI** (instead of Flask/Django) | Automatic OpenAPI generation, async‑first, excellent Pydantic integration for request validation. |
| **Raw SQL via psycopg** | Keeps the data‑access layer explicit, helps developers understand the exact queries that hit PostgreSQL, and avoids the “black‑box” of an ORM. |
| **TensorFlow `model.h5` loaded once at app startup** | Reduces latency by avoiding per‑request model reloads; the model lives in a singleton service object. |
| **JWT authentication** | Stateless tokens work well with containerised deployments and make the API horizontally scalable. |
| **Role‑based access (USER / ADMIN)** | Allows admin‑only endpoints for model management and audit logs while protecting patient data. |
| **Connection pooling (`psycopg_pool`)** | Reuses PostgreSQL connections across requests, essential for low‑latency inference services. |
| **Alembic migrations** | Even though the app uses raw SQL, schema versioning is still needed for repeatable deployments. |
| **Docker + Docker‑Compose** | Guarantees reproducible dev / CI environments and simplifies deployment to any container‑orchestrated platform. |
| **Separate service & repository layers** | Service layer owns business rules (e.g., rate‑limiting per user, image size checks) and transaction boundaries; repository layer is pure SQL. |
| **Comprehensive test suite (pytest)** | Unit tests for services, integration tests for routers, and end‑to‑end tests that hit an in‑memory PostgreSQL instance. |

---  

## Architecture  

```mermaid
flowchart TD
    Client["Client / Front‑end (web / mobile)"]
    Router["FastAPI Router"]
    Schema["Pydantic Schemas"]
    Auth["JWT Auth Dependency"]
    Service["Service Layer"]
    Model["TensorFlow Model Service"]
    Repository["Repository Layer"]
    Pool["PostgreSQL Connection Pool"]
    DB[("PostgreSQL")]

    Client --> Router
    Router --> Schema
    Router --> Auth
    Router --> Service
    Service --> Model
    Service --> Repository
    Repository --> Pool
    Pool --> DB
```

* **Router** – HTTP routing, status‑code handling, request validation.  
* **Service** – Business logic (image validation, rate‑limiting, transaction management).  
* **Model Service** – Holds the loaded TensorFlow model, performs inference, caches predictions.  
* **Repository** – Executes raw SQL for persisting users, predictions, audit logs.  
* **Pool** – Reusable connections to PostgreSQL.  

---  

## Request Flow  

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Router
    participant A as Auth (JWT)
    participant S as Service
    participant M as Model Service
    participant Repo as Repository
    participant DB as PostgreSQL

    C->>R: POST /api/v1/predict (image)
    R->>A: Validate JWT
    A->>R: user_id
    R->>S: predict(image, user_id)
    S->>M: run_inference(image)
    M-->>S: prediction
    S->>Repo: INSERT prediction record
    Repo->>DB: INSERT ...
    DB-->>Repo: OK
    Repo-->>S: persisted
    S-->>R: {prediction, confidence}
    R-->>C: JSON response
```

All steps are wrapped in a single **transaction**; if the inference fails, the DB write is rolled back, ensuring no partially persisted data.

---  

## Technology Stack  

| Technology | Purpose |
|---|---|
| **Python 3.13** | Language |
| **FastAPI** | ASGI web framework |
| **Pydantic** | Request/response validation |
| **pydantic‑settings** | Environment‑based configuration |
| **TensorFlow 2.x** | Load and run the `.h5` CNN model |
| **PostgreSQL 16** | Relational data store |
| **psycopg 3** | PostgreSQL driver |
| **psycopg‑pool** | Connection pooling |
| **PyJWT** | JWT creation/validation |
| **pwdlib + Argon2** | Secure password hashing |
| **Alembic** | Database migrations |
| **pytest** | Test runner |
| **Docker** | Containerisation |
| **Docker‑Compose** | Multi‑service orchestration |
| **Uvicorn** | ASGI server |

---  

## Project Structure  

```text
breast-cancer-api/
│
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Settings (pydantic‑settings)
│   ├── security.py            # JWT helpers, password hashing
│   ├── dependencies.py        # Common DI (e.g., DB pool, auth)
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   └── database.py        # Connection pool creator
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── tf_model.py        # Loads `model.h5` once, inference API
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── users.py
│   │   └── predictions.py
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── users.py
│   │   └── predict.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── users.py
│   │   └── predict.py
│   │
│   └── repositories/
│       ├── __init__.py
│       ├── users.py
│       └── predictions.py
│
├── migrations/
│   ├── env.py
│   └── versions/
│       └── <timestamp>_initial_schema.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_predict.py
│   └── test_health.py
│
├── Dockerfile
├── .dockerignore
├── compose.yaml
├── alembic.ini
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

*The layout mirrors the expense‑tracker project, but adds a dedicated `models/` package for the TensorFlow model.*

---  

## Database Design  

Three core tables store user information, prediction requests, and audit logs.

```mermaid
erDiagram
    USERS ||--o{ PREDICTIONS : makes
    USERS {
        UUID id PK
        TEXT email UK
        TEXT password_hash
        TEXT role
        TIMESTAMPTZ created_at
    }

    PREDICTIONS {
        UUID id PK
        UUID user_id FK
        BYTEA image_blob
        INTEGER prediction
        REAL confidence
        TEXT image_hash
        TIMESTAMPTZ created_at
    }

    AUDIT_LOGS {
        UUID id PK
        UUID user_id FK
        TEXT action
        TEXT detail
        TIMESTAMPTZ created_at
    }
```

### Users  

* `email` is unique & validated.  
* `role` is limited to `USER` or `ADMIN`.  
* Passwords are stored only as Argon2 hashes.

### Predictions  

* `image_blob` retains the original file for compliance (optional, can be disabled via config).  
* `prediction` stores the integer class; `confidence` is a float in `[0, 1]`.  
* `image_hash` (SHA‑256) enables deduplication and quick look‑ups.

### Audit Logs  

* Every privileged action (model upload, user role change, bulk deletion) writes an entry, supporting traceability required in medical environments.

---  

## Authentication & Authorization  

* **JWT** – Stateless tokens signed with `JWT_SECRET_KEY`.  
* **Roles** – `USER` can upload images & view own predictions; `ADMIN` can list all predictions, manage the model file, and view audit logs.  

All protected routes use a **dependency** that extracts the token, validates expiration, and injects `current_user` (including `role`).  

---  

## API Endpoints  

All versioned endpoints are under `/api/v1`.

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Create a new user | ❌ |
| `POST` | `/api/v1/auth/login` | Obtain JWT | ❌ |
| `POST` | `/api/v1/predict` | Upload mammography image, receive prediction | ✅ (`USER`/`ADMIN`) |
| `GET` | `/api/v1/predictions` | List own predictions (paginated) | ✅ (`USER`) |
| `GET` | `/api/v1/predictions/{id}` | Retrieve a single prediction | ✅ (`USER` owns / `ADMIN`) |
| `GET` | `/api/v1/admin/predictions` | List *all* predictions | ✅ (`ADMIN`) |
| `GET` | `/api/v1/admin/audit-logs` | View audit trail | ✅ (`ADMIN`) |
| `POST` | `/api/v1/admin/model/upload` | Replace the TensorFlow `.h5` model (admin only) | ✅ (`ADMIN`) |

### Request / Response Schemas  

* **`PredictionCreate`** – `file: UploadFile` (image), optional `metadata: dict`.
* **`PredictionResponse`** – `{prediction: int, confidence: float, timestamp: str, image_url: str}`.
* **`UserCreate` / `UserResponse`** – standard email / token fields.

---  

## Health & Readiness Checks  

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness – confirms the process is up. |
| `GET /ready` | Readiness – validates DB connectivity and that the TensorFlow model is loaded. |

Both are deliberately **outside** `/api/v1` so that orchestration tools can probe them without authentication.

---  

## Configuration & Environment Variables  

All secrets & tunables are read from the environment via **pydantic‑settings**.

```env
# Database
DATABASE_URL=postgresql://bc_user:YOUR_PASSWORD@localhost:5432/bc_db

# JWT
JWT_SECRET_KEY=replace_with_secure_random_string
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Model handling
MODEL_PATH=/app/model/model.h5          # Path inside the container
ALLOWED_IMAGE_TYPES=png,jpeg,dicom
MAX_IMAGE_SIZE_MB=10

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=INFO
```

A template exists at **`.env.example`**.  

---  

## Docker & Docker‑Compose  

**Dockerfile** builds a slim `python:3.13-slim` image, installs runtime dependencies, copies the compiled model (`model.h5`) into `/app/model/`, and starts `uvicorn`.  

**compose.yaml** spins up two services:

* `api` – the FastAPI container.
* `postgres` – a PostgreSQL 16 container (optional; you can also use a host‑installed DB).  

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env.docker
    depends_on: [postgres]
    volumes:
      - ./model:/app/model:ro   # bind‑mount the model directory
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: bc_user
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: bc_db
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

*The model file is **read‑only** inside the container; an admin can replace it via the `/admin/model/upload` endpoint, which writes to a mounted volume.*

---  

## Testing  

* **`pytest`** runs the full suite (≈ 40 tests) against a temporary PostgreSQL database created by the fixture in `tests/conftest.py`.  
* Tests cover:  
  * Authentication (register/login, token validation)  
  * Prediction endpoint (valid‑image, oversized image, unsupported type)  
  * Role‑based access control (admin vs. user)  
  * Health/readiness probes  
  * Model upload flow (admin only)  

Run tests locally:

```bash
pytest
# or with verbose output
pytest -v
```

All tests currently pass:

```text
40 passed
```

---  

## Running Locally  

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd breast-cancer-api
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt          # runtime
   pip install -r requirements-dev.txt      # testing & linting
   ```

4. **Provide a TensorFlow model**

   Place `model.h5` inside `model/` (or set `MODEL_PATH` to another location).

5. **Configure environment variables**

   ```bash
   cp .env.example .env
   # edit .env with your DB credentials and secret keys
   ```

6. **Start PostgreSQL** (local instance or Docker).

7. **Run migrations**

   ```bash
   alembic upgrade head
   ```

8. **Start the API**

   ```bash
   uvicorn app.main:app --reload
   ```

   The service will be reachable at `http://localhost:8000`.  
   Swagger UI: `http://localhost:8000/docs`  
   ReDoc: `http://localhost:8000/redoc`

---  

## Running with Docker Compose  

1. **Create Docker‑specific env file**

   ```bash
   cp .env .env.docker
   ```

2. **Build and start containers**

   ```bash
   docker compose up --build
   ```

3. **Run migrations inside the API container**

   ```bash
   docker compose run --rm api alembic upgrade head
   ```

4. **Verify**

   ```bash
   curl http://localhost:8000/ready
   # → {"status":"ready"}
   ```

---  

## Future Improvements  

| Area | Planned Enhancement |
|------|----------------------|
| **Pagination & Filtering** | `GET /predictions` with date ranges, confidence thresholds. |
| **Batch Inference** | Accept zip files / multi‑view studies and process them asynchronously (Celery + RabbitMQ). |
| **Model Versioning** | Store multiple `.h5` files, expose `/admin/model/activate` to switch versions without downtime. |
| **Explainability** | Return Grad‑CAM heatmaps alongside predictions for clinician review. |
| **Rate Limiting** | Per‑user request caps to protect GPU/CPU resources. |
| **TLS & Reverse Proxy** | Deploy behind Nginx‑Proxy‑Manager with automatic Let's Encrypt certs. |
| **CI/CD Pipeline** | GitHub Actions: lint → test → build → push Docker image. |
| **Audit Log Export** | Stream audit logs to an external SIEM (e.g., Elastic). |
| **FHIR Integration** | Emit predictions as FHIR Observation resources for EMR consumption. |
| **GPU Support** | Detect and utilise CUDA‑enabled Docker images for faster inference. |

---  

## License  

This project is released under the **MIT License**. See the `LICENSE` file for details.

---  

## Author  

**Dexter Mtetwa** – Software engineering graduate focused on Python backend development, database engineering, and applied AI/ML pipelines for the healthcare domain.  

---  

*Built with ❤️, rigorous engineering, and an eye toward clinical‑grade reliability.*
