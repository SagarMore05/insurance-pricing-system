import logging

from fastapi import APIRouter, HTTPException, status

from src.auth.schemas import LoginRequest, TokenResponse
from src.auth.security import create_access_token, verify_credentials

logger = logging.getLogger("auth")

router = APIRouter(tags=["Auth"])


@router.post("/admin/login", response_model=TokenResponse)
async def admin_login(request: LoginRequest):
    if not verify_credentials(request.username, request.password):
        logger.warning("Failed login attempt for username: %s", request.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(username=request.username)
    logger.info("Admin login successful for: %s", request.username)
    return TokenResponse(access_token=token, token_type="bearer")
