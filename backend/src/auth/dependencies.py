from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/admin/login")


async def get_current_admin(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return payload
