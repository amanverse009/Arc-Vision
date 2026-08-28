import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import corpus

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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


def call_claude(system: str, user: str, history: list = None) -> str:
    """Named call_claude for the rest of the file to stay unchanged — actually calls
    Groq's free-tier Llama API under the hood. Supports multi-turn history."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Add it in Render's Environment tab.",
        )
    messages = [{"role": "system", "content": system}]
    for turn in (history or []):
        role = "assistant" if turn.role == "model" else "user"
        messages.append({"role": role, "content": turn.text})
    messages.append({"role": "user", "content": user})

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "max_tokens": 900},
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Unexpected Groq response: {data}")


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    text: str


class AskRequest(BaseModel):
    question: str
    lang: str = "en"  # "en" or "hi"
    history: list[ChatMessage] = []


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

    # Retrieve using the current question plus recent history, so later turns
    # in the conversation still pull relevant law even if the latest message
    # is just an answer to a clarifying question ("yes, two months ago").
    search_text = req.question
    if req.history:
        search_text = " ".join(h.text for h in req.history[-4:]) + " " + req.question
    matches = corpus.search(search_text, top_k=3)

    if not matches:
        grounding_note = "No matching source was found in the current corpus for this — say so plainly rather than guessing at a specific law."
        sources = []
    else:
        grounding_note = "Ground any specific law/article you cite in the source material below. Don't invent citations not listed here."
        sources = [{"id": m["id"], "title": m["title"]} for m in matches]

    system = f"""You are Kanoon Wala, a warm, patient AI legal guidance assistant for Indian citizens, talking with someone about a real problem.
{grounding_note}

SOURCE MATERIAL:
{format_sources(matches)}

How to behave, like a good duty lawyer would on a first call:
- If you don't yet have enough detail to give complete, specific guidance — for example you don't know the timeline, whether they've already contacted anyone (police, employer, landlord), what documents/proof they have, or their state/city (some remedies are state-specific) — ask 2-4 short, specific clarifying questions. Don't ask about things they've already told you. Keep this reply short and conversational, no headers, no long explanation yet.
- Once you have enough detail (from this message or earlier in the conversation), give a full, thorough answer with:
  1. A short empathetic acknowledgement of their situation, in your own words.
  2. The relevant constitutional articles / laws that apply, named specifically, with a plain-language sentence on what each one means for their case.
  3. A clear, numbered, step-by-step action plan — concrete things they can actually do, in the order they should do them (who to contact, what to file, what to bring/keep as evidence, realistic timelines).
  4. End with: "This is general legal information, not a substitute for a qualified lawyer."
- Never limit yourself to 1-2 lines when giving the full answer — be as thorough as the situation needs. Only the clarifying-question turn should be short.
{lang_instruction(req.lang)}"""

    answer = call_claude(system, req.question, history=req.history)
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
    return {"status": "ok", "corpus_size": len(corpus.docs), "llm_configured": GROQ_API_KEY is not None}
