# AutoML Agent - Architecture

## Overview
AutoML Agent is designed using a modular, scalable architecture. The system separates the core Machine Learning pipeline from the AI Agent and the API layer, ensuring loose coupling and high maintainability.

## Core Components

1. **API Layer (FastAPI)**:
   - Handles HTTP requests, CORS, and routing.
   - Manages authentication (JWT, Google OAuth) and authorization.
   - Interacts with the database using SQLAlchemy.

2. **AI Agent (`app/agent/`)**:
   - The "brain" of the application. It uses **LiteLLM** to communicate with the LLM (e.g., Ollama/Qwen).
   - Maintains conversation memory and context window.
   - Decides which tools to call based on the user's prompt.
   - Parses tool calls and passes them to the Executor.

3. **Core ML Pipeline (`app/core/`)**:
   - Pure Python/Scikit-learn logic. 
   - Contains modules for Data Loading, EDA, Preprocessing, Training, and Report Generation.
   - It knows nothing about the LLM or HTTP; it simply takes a DataFrame and returns results.

4. **Database (`app/db/`)**:
   - PostgreSQL managed via SQLAlchemy.
   - Stores `Users`, `Sessions`, `Messages`, `Operations`, and `Artifacts`.

5. **Notifications & Storage**:
   - Generates PDFs using WeasyPrint and Jupyter Notebooks using `nbformat`.
   - Stores artifacts locally and notifies users via Email (Resend) and Web Push.

## Data Flow
1. The user uploads a dataset via the API.
2. The agent receives a prompt and analyzes the dataset schema.
3. The agent decides on a sequence of tools (e.g., EDA -> Preprocessing -> Training).
4. The `executor.py` triggers the Core ML Pipeline.
5. The pipeline saves artifacts to the local storage and updates the database.
6. The agent returns the final response to the user.