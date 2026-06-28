from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.security import decode_token

# tokenUrl is used by OpenAPI docs "Authorize" button only
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/admin/login")


async def verify_admin_key(token: str = Depends(oauth2_scheme)) -> dict:
    """
    JWT-based admin guard — replaces the old static X-Admin-Key header check.
    All existing routers that Depends(verify_admin_key) pick up JWT auth automatically.
    Returns 401 when token is missing/expired, 403 when role is not 'admin'.
    """
    payload = decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return payload
