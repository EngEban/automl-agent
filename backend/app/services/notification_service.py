"""
Notification service for sending updates via Email.
"""
from __future__ import annotations

import logging
import resend
from app.config import settings

logger = logging.getLogger(__name__)
resend.api_key = settings.RESEND_API_KEY

def send_verification_email(email: str, token: str) -> bool:
    """Send a verification link via Resend."""
    if not settings.RESEND_API_KEY:
        logger.warning("Resend API key missing. Email not sent.")
        return False

    verification_link = f"{settings.APP_BASE_URL}/api/auth/verify-email?token={token}"
    
    html_content = f"""
    <h2>Welcome to {settings.APP_NAME}!</h2>
    <p>Please click the link below to verify your email address and activate your account:</p>
    <a href="{verification_link}" style="padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; display: inline-block;">Verify Email</a>
    <p>If you didn't create an account, you can safely ignore this email.</p>
    """
    
    try:
        resend.Emails.send({
            "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>",
            "to": email,
            "subject": settings.EMAIL_VERIFY_SUBJECT,
            "html": html_content
        })
        logger.info(f"Verification email sent to {email}")
        return True
    except Exception as exc:
        logger.error(f"Resend failed: {exc}")
        return False

def send_completion_email(email: str, session_id: str, best_model: str) -> bool:
    """Send an email notifying the user that their analysis is ready."""
    if not settings.RESEND_API_KEY:
        logger.warning("Resend API key missing. Email not sent.")
        return False

    html_content = f"""
    <h2>Your Analysis is Ready! 🎉</h2>
    <p>Great news! The AI Agent has finished analyzing your dataset.</p>
    <ul>
        <li><strong>Session ID:</strong> {session_id}</li>
        <li><strong>Best Model:</strong> {best_model}</li>
    </ul>
    <p>You can now log in to your dashboard to download your PDF report, the reproducible Jupyter Notebook, and the trained model.</p>
    <a href="{settings.FRONTEND_URL}" style="padding: 10px 20px; background: #27ae60; color: white; text-decoration: none; border-radius: 5px; display: inline-block;">Go to Dashboard</a>
    """
    
    try:
        resend.Emails.send({
            "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>",
            "to": email,
            "subject": f"Your AutoML Analysis is Ready (Session: {session_id[:8]})",
            "html": html_content
        })
        logger.info(f"Completion email sent to {email}")
        return True
    except Exception as exc:
        logger.error(f"Completion email failed: {exc}")
        return False