import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

from retrieval import corpus

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None
MODEL = "claude-sonnet-4-6"

app = FastAPI(title="Kanoon Wala API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_client():
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set on the server. Add it in Replit Secrets.",
        )


def format_sources(results):
    return "\n\n".join(
        f"[{r['title']}] {r['source_text']}" for r in results
    )


def call_claude(system: str, user: str) -> str:
    require_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


class AskRequest(BaseModel):
    question: str
    lang: str = "en"  # "en" or "hi"


class SimplifyRequest(BaseModel):
    text: str
    lang: str = "en"


def lang_instruction(lang: str) -> str:
    return (
        "Respond in simple, everyday Hindi (Devanagari script)."
        if lang == "hi"
        else "Respond in simple, everyday English."
    )


@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    matches = corpus.search(req.question, top_k=3)
    if not matches:
        grounding_note = "No matching source found in the current corpus — answer cautiously and say so."
        sources = []
    else:
        grounding_note = "Base your answer ONLY on the source material below. Cite the bracketed titles you actually used."
        sources = [{"id": m["id"], "title": m["title"]} for m in matches]

    system = f"""You are Kanoon Wala, an AI legal information assistant for Indian citizens.
A citizen describes a real-life problem in plain words, not legal terms.
{grounding_note}

SOURCE MATERIAL:
{format_sources(matches)}

Structure your answer as:
1. A one-line plain-language summary of the situation.
2. The relevant constitutional articles / laws that apply, citing them by name.
3. A likely official pathway or next step.
Always end with: "This is general legal information, not a substitute for a qualified lawyer."
Keep it under 180 words. {lang_instruction(req.lang)}"""

    answer = call_claude(system, req.question)
    return {"answer": answer, "sources": sources}


@app.get("/api/explore/search")
def explore_search(q: str = ""):
    return {"results": corpus.list_all(q)}


@app.get("/api/explore/{doc_id}")
def explore_detail(doc_id: str, lang: str = "en"):
    doc = corpus.get_by_id(doc_id)
    if not doc:
        raise HTTPException(404, "not found")

    system = f"""You are Kanoon Wala's Law & Case Explorer.
Base your explanation ONLY on this source text: {doc['source_text']}

If it is a constitutional article or act, explain: (1) what it means in plain language,
(2) why it matters, (3) a real-world example of when someone would use it.
If it is a landmark court case, explain: (1) the facts, (2) the court's reasoning, (3) the outcome.
Keep it under 160 words. {lang_instruction(lang)}"""

    answer = call_claude(system, f"Explain {doc['title']}")
    return {"id": doc["id"], "title": doc["title"], "type": doc["type"], "explanation": answer}


@app.post("/api/simplify")
def simplify(req: SimplifyRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")

    system = f"""You are Kanoon Wala's Document Simplifier.
The user will paste a legal document, judgment excerpt, notice, or clause.
Break it down into: (1) a one-line plain summary, (2) key points/obligations as a short bullet list,
(3) anything the person should be careful about or act on.
Keep it under 180 words. {lang_instruction(req.lang)}"""

    answer = call_claude(system, req.text)
    return {"simplified": answer}


@app.get("/api/health")
def health():
    return {"status": "ok", "corpus_size": len(corpus.docs), "llm_configured": client is not None}
