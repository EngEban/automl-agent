# 🤖 AutoML Agent

An intelligent, autonomous AI Agent that automates the entire Data Science lifecycle. Upload a dataset, and the agent will perform EDA, clean the data, train multiple models, select the best one, and generate a professional PDF report along with a reproducible Jupyter Notebook.

## ✨ Features
- **Autonomous Agent:** Powered by LLMs (via Ollama, OpenAI, or Groq) using function calling to decide the best analysis path.
- **Smart Preprocessing:** Automatically handles missing values, categorical encoding, and feature scaling.
- **AutoML Pipeline:** Trains and compares multiple models (Random Forest, Gradient Boosting, etc.).
- **Rich Reporting:** Generates a polished PDF report with insights, charts, and model comparisons.
- **Reproducible Code:** Generates a ready-to-run `.ipynb` Jupyter Notebook.
- **Secure Authentication:** JWT-based authentication with Google OAuth and Email Verification.
- **Notifications:** Real-time Web Push notifications and Email alerts upon completion.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python 3.11, SQLAlchemy, PostgreSQL.
- **AI/Agent:** LiteLLM, Ollama (Local LLM), LangChain concepts.
- **Machine Learning:** Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Authentication:** JWT, Passlib, Python-Jose, Google OAuth2.
- **Reporting:** WeasyPrint (PDF), nbformat (Jupyter).
- **Frontend:** React, Vite, Tailwind CSS *(Coming soon)*.

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
- Python 3.11+
- PostgreSQL (or Docker)
- Ollama (for local LLM) or an API key for Groq/OpenAI.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/automl-agent.git
cd automl-agent/backend

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Environment Variables
Copy .env.example to .env and fill in your credentials (Database URL, LLM API keys, Resend API key, etc.).

4. Database Setup
bash
docker run --name automl-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=automl -p 5433:5432 -d postgres:16-alpine
5. Run the Application
bash
uvicorn app.main:app --reload --port 8100
Visit http://localhost:8100/docs to access the Swagger API documentation.

📖 Documentation
Architecture

Security