from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import os

# secret key used to sign and verify tokens — set in .env
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "changeme")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 30))

# tells FastAPI where to look for the token in the request
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def create_token(username: str) -> str:
    """Create a signed JWT token valid for EXPIRE_MINUTES."""
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(minutes=EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """
    Verify the token signature and expiry.
    Returns username if valid, raises 401 if not.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Dependency injected into protected routes via Depends().
    FastAPI calls this automatically on every /chat request.
    """
    return verify_token(token)
