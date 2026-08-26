"""
Hybrid retrieval: pgvector cosine similarity + Postgres full-text keyword search,
merged and re-ranked. This is what grounds the LLM's answers in real legal text
instead of letting it hallucinate.
"""
from sqlalchemy import select, func, text as sqltext
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LegalSourceChunk, LegalSource
from app.rag.embeddings import embed_text


async def vector_search(db: AsyncSession, query: str, top_k: int = 8) -> list[LegalSourceChunk]:
    query_embedding = embed_text(query)
    stmt = (
        select(LegalSourceChunk)
        .order_by(LegalSourceChunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def keyword_search(db: AsyncSession, query: str, top_k: int = 8) -> list[LegalSourceChunk]:
    # Postgres full-text search (to_tsvector/plainto_tsquery) as the keyword leg of the hybrid search.
    stmt = (
        select(LegalSourceChunk)
        .where(sqltext("to_tsvector('english', content) @@ plainto_tsquery('english', :q)"))
        .params(q=query)
        .limit(top_k)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def reciprocal_rank_fusion(
    result_lists: list[list[LegalSourceChunk]], k: int = 60
) -> list[LegalSourceChunk]:
    """Merge multiple ranked lists (vector + keyword) into one ranking."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, LegalSourceChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            cid = str(chunk.id)
            chunks_by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [chunks_by_id[cid] for cid in ranked_ids]


async def hybrid_retrieve(
    db: AsyncSession, query: str, top_k: int = 6
) -> list[tuple[LegalSourceChunk, LegalSource]]:
    vec_results, kw_results = await vector_search(db, query, top_k=10), await keyword_search(db, query, top_k=10)
    fused = reciprocal_rank_fusion([vec_results, kw_results])[:top_k]

    output: list[tuple[LegalSourceChunk, LegalSource]] = []
    for chunk in fused:
        source_result = await db.execute(select(LegalSource).where(LegalSource.id == chunk.source_id))
        source = source_result.scalar_one_or_none()
        if source and source.is_verified:  # never surface unverified/unreviewed sources to users
            output.append((chunk, source))
    return output
