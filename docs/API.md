# AutoML Agent - API Documentation

The AutoML Agent API is built with FastAPI. You can access the interactive Swagger UI at `http://localhost:8100/docs`.

## Authentication (`/api/auth`)
- `POST /register`: Register a new user (sends a verification email).
- `POST /login`: Authenticate and receive a JWT access token.
- `POST /google`: Authenticate using a Google OAuth2 authorization code.
- `GET /me`: Get the current user's profile.
- `POST /push-subscription`: Save a Web Push subscription for notifications.

*Note: Most endpoints require the `Authorization: Bearer <token>` header.*

## Sessions (`/api/sessions`)
- `POST /upload`: Upload a CSV/Excel/JSON dataset to start a new session.
- `POST /`: Create a session from a direct URL.
- `GET /`: List all sessions for the authenticated user.
- `GET /{session_id}`: Get details of a specific session.
- `DELETE /{session_id}`: Delete a session and its associated files.

## Chat (`/api/chat`)
- `POST /{session_id}`: Send a message to the AI Agent. The agent will execute tools and return a response.
- `GET /{session_id}/messages`: Retrieve the chat history for a session.

## Artifacts (`/api/artifacts`)
- `GET /{session_id}/report`: Download the generated PDF report.
- `GET /{session_id}/notebook`: Download the reproducible Jupyter Notebook.
- `GET /{session_id}/dataset`: Download the original dataset.