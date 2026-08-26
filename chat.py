from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import User, ChatSession, ChatMessage, MessageRole, AdminAuditLog
from app.rag.guidance import generate_guidance
from app.rag.voice import transcribe_audio, synthesize_speech
from app.schemas import (
    ChatSessionCreate, ChatSessionOut, ChatQueryRequest, ChatQueryResponse, CitedSource,
    VoiceChatQueryResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionOut)
async def create_session(
    payload: ChatSessionCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = ChatSession(user_id=user.id, language=payload.language, input_mode=payload.input_mode)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc())
    )
    return list(result.scalars().all())


@router.post("/query", response_model=ChatQueryResponse)
async def query_chat(
    payload: ChatQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.id == payload.session_id, ChatSession.user_id == user.id)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    language = payload.language or session.language

    user_msg = ChatMessage(session_id=session.id, role=MessageRole.user, content=payload.message)
    db.add(user_msg)
    await db.flush()

    guidance = await generate_guidance(db, user_problem=payload.message, language=language)

    assistant_msg = ChatMessage(
        session_id=session.id,
        role=MessageRole.assistant,
        content=guidance["reply"],
        cited_source_ids=[s["source_id"] for s in guidance["cited_sources"]],
        confidence=guidance["confidence"],
    )
    db.add(assistant_msg)
    db.add(
        AdminAuditLog(
            actor_user_id=user.id,
            event_type="chat_query",
            event_meta={"session_id": str(session.id), "confidence": guidance["confidence"]},
        )
    )
    await db.commit()
    await db.refresh(assistant_msg)

    return ChatQueryResponse(
        message_id=assistant_msg.id,
        reply=guidance["reply"],
        cited_sources=[CitedSource(**s) for s in guidance["cited_sources"]],
        confidence=guidance["confidence"],
    )


@router.post("/voice-query", response_model=VoiceChatQueryResponse)
async def voice_query_chat(
    session_id: str = Form(...),
    language: str | None = Form(None),
    audio: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Voice-first flow: user speaks -> transcribe -> run the same source-grounded
    guidance engine as text chat -> speak the reply back. This is what makes the
    platform usable for people with limited literacy, per the pitch deck.
    """
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    audio_bytes = await audio.read()
    transcript, detected_language = transcribe_audio(audio_bytes, filename_hint=audio.filename or "audio")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not understand the audio, please try again")

    # Prefer an explicit language param, then the session's language, then whisper's detection.
    response_language = language or session.language or ("hi" if detected_language.startswith("hi") else "en")

    user_msg = ChatMessage(
        session_id=session.id, role=MessageRole.user, content=transcript, audio_url=None
    )
    db.add(user_msg)
    await db.flush()

    guidance = await generate_guidance(db, user_problem=transcript, language=response_language)
    reply_audio_url = await synthesize_speech(guidance["reply"], language=response_language)

    assistant_msg = ChatMessage(
        session_id=session.id,
        role=MessageRole.assistant,
        content=guidance["reply"],
        audio_url=reply_audio_url,
        cited_source_ids=[s["source_id"] for s in guidance["cited_sources"]],
        confidence=guidance["confidence"],
    )
    db.add(assistant_msg)
    db.add(
        AdminAuditLog(
            actor_user_id=user.id,
            event_type="voice_chat_query",
            event_meta={"session_id": str(session.id), "confidence": guidance["confidence"]},
        )
    )
    await db.commit()
    await db.refresh(assistant_msg)

    return VoiceChatQueryResponse(
        message_id=assistant_msg.id,
        reply=guidance["reply"],
        cited_sources=[CitedSource(**s) for s in guidance["cited_sources"]],
        confidence=guidance["confidence"],
        transcript=transcript,
        reply_audio_url=reply_audio_url,
    )
