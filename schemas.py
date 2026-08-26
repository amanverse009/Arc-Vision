import uuid
from datetime import datetime
from pydantic import BaseModel, Field


# ---- Auth ----
class OTPSendRequest(BaseModel):
    phone_number: str = Field(..., examples=["+919876543210"])


class OTPVerifyRequest(BaseModel):
    phone_number: str
    otp_code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    is_new_user: bool


class UserOut(BaseModel):
    id: uuid.UUID
    phone_number: str
    full_name: str | None
    preferred_language: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---- Chat / RAG guidance ----
class ChatSessionCreate(BaseModel):
    language: str = "hi"
    input_mode: str = "text"


class ChatSessionOut(BaseModel):
    id: uuid.UUID
    title: str | None
    language: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatQueryRequest(BaseModel):
    session_id: uuid.UUID
    message: str
    language: str | None = None


class CitedSource(BaseModel):
    source_id: uuid.UUID
    title: str
    citation: str | None = None
    excerpt: str


class ChatQueryResponse(BaseModel):
    message_id: uuid.UUID
    reply: str
    cited_sources: list[CitedSource]
    confidence: str
    disclaimer: str = (
        "This is AI-generated legal information, not a substitute for a qualified lawyer."
    )


class VoiceChatQueryResponse(ChatQueryResponse):
    transcript: str
    reply_audio_url: str


# ---- Law & Case Explorer ----
class LegalSourceSearchResult(BaseModel):
    id: uuid.UUID
    source_type: str
    title: str
    citation: str | None
    snippet: str

    class Config:
        from_attributes = True


class LegalSourceDetail(BaseModel):
    id: uuid.UUID
    source_type: str
    title: str
    citation: str | None
    full_text: str
    simplified_text: str | None
    summary_json: dict | None
    source_url: str | None

    class Config:
        from_attributes = True


# ---- Document Simplifier ----
class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    status: str


class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    simplified_summary: str | None


# ---- Admin ----
class AdminStatsResponse(BaseModel):
    total_users: int
    total_chat_sessions: int
    total_documents_processed: int
    total_verified_sources: int
    total_unverified_sources: int


class AdminAuditLogOut(BaseModel):
    id: uuid.UUID
    event_type: str
    event_meta: dict | None
    created_at: datetime

    class Config:
        from_attributes = True
