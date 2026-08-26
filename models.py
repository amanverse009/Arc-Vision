import uuid
import enum
from datetime import datetime

from sqlalchemy import (
    String, Boolean, DateTime, ForeignKey, Text, Integer, Enum, func, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.database import Base

EMBEDDING_DIM = 768  # matches paraphrase-multilingual-mpnet-base-v2


def gen_uuid():
    return uuid.uuid4()


class UserRole(str, enum.Enum):
    citizen = "citizen"
    admin = "admin"


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    phone_number: Mapped[str] = mapped_column(String(15), unique=True, index=True, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    preferred_language: Mapped[str] = mapped_column(String(10), default="hi")
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.citizen)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")
    documents: Mapped[list["UserDocument"]] = relationship(back_populates="user")


class OTPRequest(Base):
    """Short-lived OTP codes for phone login. One row per OTP send."""
    __tablename__ = "otp_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    phone_number: Mapped[str] = mapped_column(String(15), index=True, nullable=False)
    otp_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_used: Mapped[bool] = mapped_column(Boolean, default=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# LAW / CASE EXPLORER  (source-grounded knowledge base, RAG-indexed)
# ---------------------------------------------------------------------------
class SourceType(str, enum.Enum):
    constitution_article = "constitution_article"
    act_section = "act_section"
    court_case = "court_case"
    nalsa_resource = "nalsa_resource"


class LegalSource(Base):
    """One authoritative legal document/provision (article, act section, judgment, etc.)."""
    __tablename__ = "legal_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)         # e.g. "Article 21"
    citation: Mapped[str | None] = mapped_column(String(200), nullable=True)  # e.g. "AIR 1978 SC 597"
    full_text: Mapped[str] = mapped_column(Text, nullable=False)             # original authoritative text
    simplified_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # AI-simplified, cached
    simplified_text_hi: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # facts/issues/reasoning/outcome
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)  # admin-verified before shown to users
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks: Mapped[list["LegalSourceChunk"]] = relationship(back_populates="source", cascade="all, delete-orphan")


class LegalSourceChunk(Base):
    """Chunked + embedded pieces of a LegalSource, used for vector retrieval (RAG)."""
    __tablename__ = "legal_source_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("legal_sources.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    source: Mapped["LegalSource"] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# CHAT / PROBLEM-TO-LAW GUIDANCE (the core RAG guidance flow)
# ---------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    input_mode: Mapped[str] = mapped_column(String(10), default="text")  # "text" | "voice"
    language: Mapped[str] = mapped_column(String(10), default="hi")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    audio_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # For assistant messages: which legal sources were retrieved/cited, and a confidence flag
    cited_source_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "high" | "medium" | "low"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


# ---------------------------------------------------------------------------
# DOCUMENT SIMPLIFIER
# ---------------------------------------------------------------------------
class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    simplified = "simplified"
    failed = "failed"


class UserDocument(Base):
    __tablename__ = "user_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    original_filename: Mapped[str] = mapped_column(String(300))
    storage_path: Mapped[str] = mapped_column(String(500))
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    simplified_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(Enum(DocumentStatus), default=DocumentStatus.uploaded)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="documents")


# ---------------------------------------------------------------------------
# ADMIN / AUDIT
# ---------------------------------------------------------------------------
class AdminAuditLog(Base):
    """Every notable platform event, so the admin panel can monitor activity."""
    __tablename__ = "admin_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=gen_uuid)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(100))  # e.g. "otp_sent", "chat_query", "doc_uploaded", "source_verified"
    event_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
