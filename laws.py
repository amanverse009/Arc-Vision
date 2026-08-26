from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import LegalSource
from app.schemas import LegalSourceSearchResult, LegalSourceDetail

router = APIRouter(prefix="/laws", tags=["law-and-case-explorer"])


@router.get("/search", response_model=list[LegalSourceSearchResult])
async def search_laws(
    q: str = Query(..., min_length=2),
    source_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(LegalSource).where(
        LegalSource.is_verified.is_(True),
        or_(LegalSource.title.ilike(f"%{q}%"), LegalSource.citation.ilike(f"%{q}%")),
    )
    if source_type:
        stmt = stmt.where(LegalSource.source_type == source_type)
    stmt = stmt.limit(20)

    result = await db.execute(stmt)
    sources = result.scalars().all()
    return [
        LegalSourceSearchResult(
            id=s.id,
            source_type=s.source_type.value,
            title=s.title,
            citation=s.citation,
            snippet=(s.simplified_text or s.full_text)[:200],
        )
        for s in sources
    ]


@router.get("/{source_id}", response_model=LegalSourceDetail)
async def get_law_detail(source_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(LegalSource).where(LegalSource.id == source_id, LegalSource.is_verified.is_(True))
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Not found")
    return LegalSourceDetail(
        id=source.id,
        source_type=source.source_type.value,
        title=source.title,
        citation=source.citation,
        full_text=source.full_text,
        simplified_text=source.simplified_text,
        summary_json=source.summary_json,
        source_url=source.source_url,
    )
