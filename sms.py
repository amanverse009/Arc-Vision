"""
SMS/OTP delivery. Provider is chosen via settings.SMS_PROVIDER:
  - "dev"    -> logs the OTP instead of sending (default, safe for local dev)
  - "twilio" -> sends via Twilio Programmable Messaging
  - "msg91"  -> sends via MSG91 (popular India-focused provider, cheaper for Indian numbers)

You still need your own account with whichever provider you pick — that part can't
be automated for you, but the integration code itself is complete and ready to use
as soon as you drop credentials into .env.
"""
import logging

import httpx

from app.config import settings

logger = logging.getLogger("kanoon_wala.sms")


async def _send_via_twilio(phone_number: str, otp_code: str) -> bool:
    from twilio.rest import Client

    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Your Kanoon Wala verification code is {otp_code}. It expires in "
             f"{settings.OTP_EXPIRY_SECONDS // 60} minutes. Do not share this code.",
        from_=settings.TWILIO_FROM_NUMBER,
        to=phone_number,
    )
    return message.sid is not None


async def _send_via_msg91(phone_number: str, otp_code: str) -> bool:
    # MSG91's OTP API expects the number without a leading "+".
    mobile = phone_number.lstrip("+")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://control.msg91.com/api/v5/otp",
            headers={"authkey": settings.MSG91_AUTH_KEY, "Content-Type": "application/json"},
            json={
                "mobile": mobile,
                "otp": otp_code,
                "sender": settings.MSG91_SENDER_ID,
                "template_id": settings.MSG91_TEMPLATE_ID or None,
            },
        )
        return resp.status_code == 200


async def send_otp_sms(phone_number: str, otp_code: str) -> bool:
    provider = settings.SMS_PROVIDER.lower()

    if provider == "dev":
        logger.info(f"[DEV OTP] {phone_number} -> {otp_code}")
        return True

    try:
        if provider == "twilio":
            return await _send_via_twilio(phone_number, otp_code)
        if provider == "msg91":
            return await _send_via_msg91(phone_number, otp_code)
    except Exception:
        logger.exception(f"Failed to send OTP via {provider} to {phone_number}")
        return False

    raise ValueError(f"Unknown SMS_PROVIDER: {provider!r} (expected dev, twilio, or msg91)")
