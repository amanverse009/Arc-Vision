from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import require_admin
from app.database import get_db
from app.models import (
    User, ChatSession, UserDocument, DocumentStatus, LegalSource, AdminAuditLog,
)
from app.rag.embeddings import chunk_text, embed_batch
from app.models import LegalSourceChunk
from app.schemas import AdminStatsResponse, AdminAuditLogOut

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/stats", response_model=AdminStatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    total_sessions = (await db.execute(select(func.count(ChatSession.id)))).scalar_one()
    total_docs = (
        await db.execute(
            select(func.count(UserDocument.id)).where(UserDocument.status == DocumentStatus.simplified)
        )
    ).scalar_one()
    total_verified = (
        await db.execute(select(func.count(LegalSource.id)).where(LegalSource.is_verified.is_(True)))
    ).scalar_one()
    total_unverified = (
        await db.execute(select(func.count(LegalSource.id)).where(LegalSource.is_verified.is_(False)))
    ).scalar_one()

    return AdminStatsResponse(
        total_users=total_users,
        total_chat_sessions=total_sessions,
        total_documents_processed=total_docs,
        total_verified_sources=total_verified,
        total_unverified_sources=total_unverified,
    )


@router.get("/audit-logs", response_model=list[AdminAuditLogOut])
async def get_audit_logs(limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AdminAuditLog).order_by(AdminAuditLog.created_at.desc()).limit(limit))
    return list(result.scalars().all())


@router.post("/legal-sources/{source_id}/verify")
async def verify_legal_source(source_id: str, db: AsyncSession = Depends(get_db)):
    """
    Admin sign-off gate: a LegalSource only becomes visible to users / usable by
    the RAG engine (see is_verified filters in retrieval.py and laws.py) after
    an admin approves it here. Keeps unreviewed/uncurated content out of guidance.
    """
    result = await db.execute(select(LegalSource).where(LegalSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        return {"error": "not found"}
    source.is_verified = True
    await db.commit()
    return {"status": "verified", "source_id": source_id}


@router.post("/legal-sources/{source_id}/reindex")
async def reindex_legal_source(source_id: str, db: AsyncSession = Depends(get_db)):
    """Re-chunk and re-embed a source's full_text — call after editing content."""
    result = await db.execute(select(LegalSource).where(LegalSource.id == source_id))
    source = result.scalar_one_or_none()
    if not source:
        return {"error": "not found"}

    await db.execute(
        LegalSourceChunk.__table__.delete().where(LegalSourceChunk.source_id == source.id)
    )

    chunks = chunk_text(source.full_text)
    embeddings = embed_batch(chunks)
    for idx, (content, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(LegalSourceChunk(source_id=source.id, chunk_index=idx, content=content, embedding=embedding))

    await db.commit()
    return {"status": "reindexed", "chunk_count": len(chunks)}
