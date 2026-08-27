import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import corpus

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

app = FastAPI(title="Kanoon Wala API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Kanoon Wala API is running"}


def call_claude(system: str, user: str) -> str:
    """Named call_claude for the rest of the file to stay unchanged — actually calls
    Google Gemini's free-tier API under the hood."""
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not set on the server. Add it in Render's Environment tab.",
        )
    resp = requests.post(
        GEMINI_URL,
        params={"key": GOOGLE_API_KEY},
        json={
            "contents": [{"parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": 700},
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Unexpected Gemini response: {data}")


class AskRequest(BaseModel):
    question: str
    lang: str = "en"  # "en" or "hi"


class SimplifyRequest(BaseModel):
    text: str
    lang: str = "en"


def format_sources(results):
    return "\n\n".join(
        f"[{r['title']}] {r['source_text']}" for r in results
    )


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
    return {"status": "ok", "corpus_size": len(corpus.docs), "llm_configured": GOOGLE_API_KEY is not None}
