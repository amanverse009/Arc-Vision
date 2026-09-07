import os
import re

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval import corpus

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
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


# ---------------------------------------------------------------------------
# "Law changed?" detector — flags old IPC/CrPC/Evidence Act references and
# points to the current BNS/BNSS/BSA equivalent. Pure pattern matching on our
# own verified corpus notes, no AI guessing involved.
# ---------------------------------------------------------------------------
OLD_LAW_PATTERNS = [
    (re.compile(r"\bIPC\b", re.I), "IPC"),
    (re.compile(r"\bIndian Penal Code\b", re.I), "IPC"),
    (re.compile(r"\bCrPC\b", re.I), "CrPC"),
    (re.compile(r"\bCode of Criminal Procedure\b", re.I), "CrPC"),
    (re.compile(r"\bIndian Evidence Act\b", re.I), "Evidence Act"),
    (re.compile(r"\bsection\s*154\b.{0,15}\b(crpc|criminal procedure)\b", re.I), "Section 154 CrPC"),
]


def detect_old_law_reference(text: str):
    hits = sorted({label for pattern, label in OLD_LAW_PATTERNS if pattern.search(text)})
    if not hits:
        return None
    note = corpus.get_by_id("bns-note")
    return {
        "detected_terms": hits,
        "warning": (
            f"This mentions the old {', '.join(hits)} framework. India replaced the IPC, CrPC and "
            "Evidence Act with the BNS, BNSS and BSA on 1 July 2024."
        ),
        "current_framework": note["source_text"] if note else None,
    }


# ---------------------------------------------------------------------------
# Emergency / immediate-danger detector. This is intentionally simple keyword
# matching, NOT an AI judgment call — for something this safety-critical, the
# response text is hard-coded and verified rather than model-generated, so it
# can never hallucinate a wrong helpline number. This is a first-pass filter:
# it will miss situations that don't use these words, and a real production
# version would need much more careful, professionally-reviewed detection.
# ---------------------------------------------------------------------------
EMERGENCY_PATTERNS = re.compile(
    r"\b(rape|raped|assault(ed)?|attack(ed|ing)?|trying to (kill|hurt|break in)|"
    r"outside my (house|door|home)|he'?s (here|outside|coming)|in danger|"
    r"domestic violence.{0,20}(now|right now|happening)|hit(ting)? me|"
    r"stabbed|weapon|gun|knife.{0,15}(me|threat)|kidnap|abduct|"
    r"बलात्कार|हमला|मार रहा|खतरे में|अभी.{0,10}बाहर)",
    re.I,
)

EMERGENCY_SAFETY_BLOCK_EN = """🚨 IMMEDIATE SAFETY FIRST

1. If you are in immediate danger, call 112 right now (India's national emergency number — police, ambulance, fire, all in one).
2. Women-specific support: 181 (Women Helpline) or 1091 (Women Police Helpline) — both work alongside 112.
3. If you can, move to a safe, locked location or somewhere with other people around.
4. Contact a trusted person nearby and tell them where you are.
5. Preserve evidence if it's safe to do so — don't wash, change, or destroy anything; save messages, photos, or messages that document what happened.
6. Get medical help as soon as possible, even if injuries seem minor — a hospital visit also creates an official record.

Once you are safe, come back here and I can walk you through the legal process (FIR, medical examination report, next steps) at your pace."""

EMERGENCY_SAFETY_BLOCK_HI = """🚨 पहले अपनी सुरक्षा

1. अगर आप तुरंत खतरे में हैं, तो अभी 112 पर कॉल करें (भारत का राष्ट्रीय आपातकालीन नंबर — पुलिस, एम्बुलेंस, फायर, सब एक में)।
2. महिलाओं के लिए विशेष सहायता: 181 (Women Helpline) या 1091 (Women Police Helpline)।
3. अगर हो सके तो किसी सुरक्षित, बंद जगह पर जाएं या लोगों के आसपास रहें।
4. किसी भरोसेमंद व्यक्ति से संपर्क करें और उन्हें बताएं कि आप कहां हैं।
5. अगर सुरक्षित हो तो सबूत सुरक्षित रखें — कुछ भी धोएं, बदलें या नष्ट न करें; जो हुआ उससे जुड़े मैसेज या फोटो सुरक्षित रखें।
6. जल्द से जल्द मेडिकल मदद लें, भले ही चोट मामूली लगे — अस्पताल जाने से एक आधिकारिक रिकॉर्ड भी बनता है।

सुरक्षित होने के बाद, यहां वापस आएं और मैं आपको कानूनी प्रक्रिया (FIR, मेडिकल जांच, अगले कदम) में आपकी गति से मदद करूंगा।"""


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

    # Emergency check runs first and short-circuits with a hard-coded,
    # verified safety response — never AI-generated for this part.
    if EMERGENCY_PATTERNS.search(req.question):
        safety_block = EMERGENCY_SAFETY_BLOCK_HI if req.lang == "hi" else EMERGENCY_SAFETY_BLOCK_EN
        return {"answer": safety_block, "sources": [], "emergency": True, "law_change_notice": None}

    old_law_notice = detect_old_law_reference(req.question)

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
        sources = [
            {"id": m["id"], "title": m["title"], "last_verified": m.get("last_verified")}
            for m in matches
        ]

    system = f"""You are Kanoon Wala, a warm, patient AI legal guidance assistant for Indian citizens, talking with someone about a real problem.
{grounding_note}

SOURCE MATERIAL:
{format_sources(matches)}

Important: India replaced the IPC, CrPC and Evidence Act with the BNS, BNSS and BSA in July 2024. Always cite the current BNS/BNSS/BSA section where you know it, not the old IPC/CrPC section — only mention the old code name in parentheses if it helps the person recognise a term they've heard before (e.g. "Section 173 BNSS (replaces the old Section 154 CrPC)").

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
    return {"answer": answer, "sources": sources, "emergency": False, "law_change_notice": old_law_notice}


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
    return {
        "id": doc["id"],
        "title": doc["title"],
        "type": doc["type"],
        "explanation": answer,
        "last_verified": doc.get("last_verified"),
    }


@app.post("/api/simplify")
def simplify(req: SimplifyRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")

    old_law_notice = detect_old_law_reference(req.text)

    system = f"""You are Kanoon Wala's Document Simplifier.
The user will paste a legal document, judgment excerpt, notice, or clause.
Break it down into: (1) a one-line plain summary, (2) key points/obligations as a short bullet list,
(3) anything the person should be careful about or act on.
If the pasted text refers to the old IPC, CrPC, or Evidence Act, mention the current BNS/BNSS/BSA
equivalent if you know it, alongside the old reference.
Keep it under 180 words. {lang_instruction(req.lang)}"""

    answer = call_claude(system, req.text)
    return {"simplified": answer, "law_change_notice": old_law_notice}


@app.get("/api/health")
def health():
    return {"status": "ok", "corpus_size": len(corpus.docs), "llm_configured": GROQ_API_KEY is not None}
