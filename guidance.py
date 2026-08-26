"""
Core "problem -> rights" guidance engine.
Takes a citizen's plain-language problem, retrieves grounding sources via
hybrid_retrieve(), and asks the LLM to answer ONLY from those sources -
returning citations and a confidence label so the frontend can show
"uncertain, please consult a lawyer" when appropriate.
"""
import json

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import LegalSourceChunk, LegalSource
from app.rag.retrieval import hybrid_retrieve

SYSTEM_PROMPT = """You are a legal information assistant for Indian citizens called Kanoon Wala.

Rules you MUST follow:
1. Answer ONLY using the SOURCES provided below. Do not use outside legal knowledge.
2. If the sources do not clearly answer the question, say so plainly and set confidence to "low".
3. Never state something as legal fact unless it is supported by a source. Cite sources by their [index].
4. Use simple, plain language a non-lawyer can understand. Avoid legal jargon; explain any term you must use.
5. You are NOT a lawyer. Always make clear this is general legal information, not legal advice,
   and suggest an official pathway (e.g. legal aid, NALSA, relevant authority) where relevant.
6. Respond in {language}.

Return your answer as strict JSON with this shape:
{{
  "reply": "<the explanation to show the user>",
  "cited_indices": [<ints referring to the SOURCES list below that you actually relied on>],
  "confidence": "high" | "medium" | "low"
}}
"""

LANGUAGE_NAMES = {"hi": "Hindi", "en": "English"}


def _build_sources_block(chunks_with_sources: list[tuple[LegalSourceChunk, LegalSource]]) -> str:
    lines = []
    for idx, (chunk, source) in enumerate(chunks_with_sources):
        lines.append(
            f"[{idx}] ({source.source_type.value}) {source.title}"
            f"{f' - {source.citation}' if source.citation else ''}\n{chunk.content}"
        )
    return "\n\n".join(lines)


async def generate_guidance(
    db: AsyncSession, user_problem: str, language: str = "hi"
) -> dict:
    grounding = await hybrid_retrieve(db, user_problem, top_k=6)

    if not grounding:
        return {
            "reply": (
                "I couldn't find a clearly relevant, verified law or case for this in my knowledge base yet. "
                "Please consult a lawyer or your nearest Legal Services Authority (NALSA) for accurate guidance."
            ),
            "cited_sources": [],
            "confidence": "low",
        }

    sources_block = _build_sources_block(grounding)
    lang_name = LANGUAGE_NAMES.get(language, "English")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT.format(language=lang_name),
        messages=[
            {
                "role": "user",
                "content": (
                    f"CITIZEN'S PROBLEM:\n{user_problem}\n\n"
                    f"SOURCES:\n{sources_block}\n\n"
                    "Respond with the JSON object only, no other text."
                ),
            }
        ],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {"reply": raw_text, "cited_indices": [], "confidence": "low"}

    cited_sources = []
    for i in parsed.get("cited_indices", []):
        if 0 <= i < len(grounding):
            chunk, source = grounding[i]
            cited_sources.append(
                {
                    "source_id": str(source.id),
                    "title": source.title,
                    "citation": source.citation,
                    "excerpt": chunk.content[:280],
                }
            )

    return {
        "reply": parsed.get("reply", ""),
        "cited_sources": cited_sources,
        "confidence": parsed.get("confidence", "medium"),
    }
