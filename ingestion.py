from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import require_admin
from app.database import get_db
from app.models import LegalSource, LegalSourceChunk, SourceType
from app.rag.embeddings import chunk_text, embed_batch

router = APIRouter(prefix="/admin/legal-sources", tags=["admin", "ingestion"], dependencies=[Depends(require_admin)])


class LegalSourceCreate(BaseModel):
    source_type: SourceType
    title: str
    citation: str | None = None
    full_text: str
    source_url: str | None = None
    auto_verify: bool = False  # keep False by default so an admin reviews before it reaches users


@router.post("")
async def create_legal_source(payload: LegalSourceCreate, db: AsyncSession = Depends(get_db)):
    source = LegalSource(
        source_type=payload.source_type,
        title=payload.title,
        citation=payload.citation,
        full_text=payload.full_text,
        source_url=payload.source_url,
        is_verified=payload.auto_verify,
    )
    db.add(source)
    await db.flush()

    chunks = chunk_text(payload.full_text)
    embeddings = embed_batch(chunks)
    for idx, (content, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(LegalSourceChunk(source_id=source.id, chunk_index=idx, content=content, embedding=embedding))

    await db.commit()
    await db.refresh(source)
    return {"id": str(source.id), "is_verified": source.is_verified, "chunk_count": len(chunks)}
