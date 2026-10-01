# RAG Project

A modular RAG application over the Egyptian Civil Code.

Project documentation will be expanded as the project progresses through each development day.

## Running with Docker

To run the RAG API service using Docker:

```bash
git clone <repository-url>
cd Egyptian-Civil-RAG
docker compose up --build
```

### Explanation

- `docker compose up --build` builds the Docker image using the `Dockerfile` and starts the `rag-api` service defined in `docker-compose.yml`.
- The container exposes the FastAPI service on `localhost:8000`.
- Environment variables (including `GOOGLE_API_KEY`) are loaded from `.env` via `docker-compose.yml`. You can copy `.env.example` to `.env` and fill in your Gemini API key before starting the service.

### Accessing the API

Once running, you can access and test the service:

- **Interactive API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **Query Endpoint**: `POST http://localhost:8000/ask` with JSON body `{"question": "your question here"}`

