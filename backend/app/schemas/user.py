from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserLogin(UserBase):
    password: str

class UserRead(UserBase):
    id: str
    is_active: bool
    push_subscription: Optional[dict] = None # <--- تم التعديل
    created_at: datetime

    model_config = {"from_attributes": True}

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead

class GoogleAuthRequest(BaseModel):
    code: str
    redirect_uri: str = "postmessage"