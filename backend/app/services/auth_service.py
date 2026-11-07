from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.db.crud import get_user_by_email, create_user
from app.db.models import User

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None

async def exchange_google_code(code: str, redirect_uri: str = "postmessage") -> dict:
    """Exchange Google Auth code for user info."""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_res = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri, # <--- تم التعديل هنا
                "grant_type": "authorization_code",
            }
        )
        token_res.raise_for_status()
        token_data = token_res.json()
        
        # Get user info
        user_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"}
        )
        user_res.raise_for_status()
        return user_res.json()

def get_or_create_google_user(db: DBSession, google_data: dict) -> User:
    email = google_data["email"]
    google_id = google_data["id"]
    
    user = get_user_by_email(db, email)
    if user:
        # Update google_id if not set
        if not user.google_id:
            user.google_id = google_id
            db.commit()
        return user
        
    # Create new user without password (OAuth only)
    return create_user(
        db,
        email=email,
        google_id=google_id,
        hashed_password=None,
        is_active=True
    )