# Local Setup Guide

## Prerequisites
- Python 3.11+
- Docker Desktop (for PostgreSQL)
- Ollama (for local LLM) OR an API key for Groq/OpenAI.

## Step 1: Clone and Install
```bash
git clone https://github.com/yourusername/automl-agent.git
cd automl-agent/backend
python -m venv .venv
# Activate venv (Windows)
.\.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
Step 2: Database Setup
Start a PostgreSQL container using Docker:

bash
docker run --name automl-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=automl -p 5433:5432 -d postgres:16-alpine
Step 3: Environment Variables
Copy .env.example to .env and fill in the required values:

DATABASE_URL

LLM_PROVIDER (e.g., ollama or groq)

LLM_BASE_URL (if using Ollama via ngrok)

RESEND_API_KEY (for emails)

GOOGLE_CLIENT_ID (for OAuth)

Step 4: Run the Backend
bash
uvicorn app.main:app --reload --port 8100
The API will be available at http://localhost:8100/docs.