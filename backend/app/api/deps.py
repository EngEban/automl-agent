"""
FastAPI dependencies.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session as DBSession

from app.db.session import get_db
from app.db.models import User
from app.services.auth_service import decode_token

DB = Annotated[DBSession, Depends(get_db)]

# التغيير هنا: استخدام HTTPBearer بدلاً من OAuth2PasswordBearer
security_scheme = HTTPBearer()

def get_current_user(
    db: DB, 
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_scheme)]
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if not payload or "sub" not in payload:
        raise credentials_exception
        
    user_id = payload["sub"]
    user = db.get(User, user_id)
    
    if user is None:
        raise credentials_exception
        
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")
        
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]