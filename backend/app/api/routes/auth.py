from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from app.api.deps import DB, CurrentUser
from app.db.crud import get_user_by_email, create_user
from app.db.models import User
from app.schemas.user import UserCreate, UserLogin, UserRead, Token, GoogleAuthRequest
from app.services import auth_service, notification_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/register", response_model=dict)
def register(payload: UserCreate, db: DB):
    if get_user_by_email(db, payload.email):
        raise HTTPException(400, "Email already registered")
    
    hashed_pw = auth_service.get_password_hash(payload.password)
    user = create_user(
        db, 
        email=payload.email, 
        hashed_password=hashed_pw,
        is_active=False
    )
    
    token = auth_service.create_access_token({"sub": user.id, "type": "verify"})
    notification_service.send_verification_email(user.email, token)
    
    return {"message": "Registration successful. Please check your email to verify."}

@router.get("/verify-email")
def verify_email(token: str, db: DB):
    payload = auth_service.decode_token(token)
    if not payload or payload.get("type") != "verify":
        raise HTTPException(400, "Invalid or expired verification token")
        
    user_id = payload.get("sub")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
        
    if not user.is_active:
        user.is_active = True
        db.commit()
        
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?verified=true")

@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: DB):
    user = get_user_by_email(db, payload.email)
    if not user or not user.hashed_password:
        raise HTTPException(401, "Invalid credentials")
    if not auth_service.verify_password(payload.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(403, "Please verify your email first")
        
    access_token = auth_service.create_access_token({"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@router.post("/google", response_model=Token)
async def google_login(payload: GoogleAuthRequest, db: DB):
    try:
        google_data = await auth_service.exchange_google_code(payload.code, payload.redirect_uri)
    except Exception as exc:
        raise HTTPException(400, f"Google auth failed: {exc}")
        
    user = auth_service.get_or_create_google_user(db, google_data)
    if not user.is_active:
        user.is_active = True
        db.commit()
        
    access_token = auth_service.create_access_token({"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": user}

@router.get("/me", response_model=UserRead)
def get_me(current_user: CurrentUser):
    return current_user

@router.post("/push-subscription")
def save_push_subscription(subscription: dict, db: DB, current_user: CurrentUser):
    """Save the browser push notification subscription."""
    current_user.push_subscription = subscription
    db.commit()
    return {"message": "Push subscription saved successfully"}
