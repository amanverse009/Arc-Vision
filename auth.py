from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import generate_otp_code, hash_otp, verify_otp, create_access_token
from app.core.sms import send_otp_sms
from app.database import get_db
from app.models import User, OTPRequest, UserRole, AdminAuditLog
from app.schemas import OTPSendRequest, OTPVerifyRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_OTP_ATTEMPTS = 5


@router.post("/send-otp", status_code=status.HTTP_200_OK)
async def send_otp(payload: OTPSendRequest, db: AsyncSession = Depends(get_db)):
    otp_code = generate_otp_code()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.OTP_EXPIRY_SECONDS)

    otp_row = OTPRequest(
        phone_number=payload.phone_number,
        otp_hash=hash_otp(otp_code),
        expires_at=expires_at,
    )
    db.add(otp_row)
    await db.commit()

    sent = await send_otp_sms(payload.phone_number, otp_code)
    if not sent:
        raise HTTPException(status_code=502, detail="Failed to send OTP, try again")

    return {"message": "OTP sent", "expires_in_seconds": settings.OTP_EXPIRY_SECONDS}


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_otp_endpoint(payload: OTPVerifyRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(OTPRequest)
        .where(OTPRequest.phone_number == payload.phone_number, OTPRequest.is_used.is_(False))
        .order_by(OTPRequest.created_at.desc())
    )
    otp_row = result.scalars().first()

    if not otp_row or otp_row.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="OTP expired or not found, request a new one")

    if otp_row.attempt_count >= MAX_OTP_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts, request a new OTP")

    if not verify_otp(payload.otp_code, otp_row.otp_hash):
        otp_row.attempt_count += 1
        await db.commit()
        raise HTTPException(status_code=400, detail="Incorrect OTP")

    otp_row.is_used = True

    # find or create the user
    user_result = await db.execute(select(User).where(User.phone_number == payload.phone_number))
    user = user_result.scalar_one_or_none()
    is_new_user = user is None

    if user is None:
        role = (
            UserRole.admin
            if settings.ADMIN_BOOTSTRAP_PHONE and payload.phone_number == settings.ADMIN_BOOTSTRAP_PHONE
            else UserRole.citizen
        )
        user = User(phone_number=payload.phone_number, role=role)
        db.add(user)
        await db.flush()

    user.last_login_at = datetime.now(timezone.utc)
    db.add(AdminAuditLog(actor_user_id=user.id, event_type="login", event_meta={"is_new_user": is_new_user}))
    await db.commit()
    await db.refresh(user)

    token = create_access_token(subject=str(user.id), role=user.role.value)
    return TokenResponse(access_token=token, user_id=user.id, is_new_user=is_new_user)
