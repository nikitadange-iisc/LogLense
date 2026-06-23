"""
Auth routes.

POST /api/auth/login  — exchange credentials for a JWT
GET  /api/auth/me     — return the currently logged-in username
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import USERS, create_token, verify_user
from dependencies import get_current_user

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/auth/login")
async def login(req: LoginRequest):
    if not verify_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": create_token(req.username), "username": req.username}


@router.get("/auth/me")
async def me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}


@router.get("/auth/users")
async def list_users(current_user: str = Depends(get_current_user)):
    """Return list of allowed usernames (no passwords)."""
    return {"users": list(USERS.keys()), "current": current_user}
