# Kanoon Wala — Backend

AI-powered legal & constitutional rights guidance platform. FastAPI + PostgreSQL/pgvector +
Anthropic Claude, built for the "Problem to Rights" flow described in the SIH pitch deck.

## Architecture

```
app/
  main.py            FastAPI app, routes registration, /health, /about
  config.py           Settings (env vars)
  database.py          Async SQLAlchemy engine/session
  models.py            All DB tables (users, OTP, legal sources + chunks, chat, documents, audit log)
  schemas.py            Pydantic request/response models
  core/
    security.py         OTP hashing + JWT
    sms.py               OTP delivery (swap in a real SMS provider)
    deps.py              get_current_user / require_admin FastAPI dependencies
  rag/
    embeddings.py        Multilingual sentence-transformer embeddings + text chunker
    retrieval.py          Hybrid (vector + keyword) search over legal_source_chunks
    guidance.py            "Problem -> rights" flow: retrieve, then ask Claude to answer
                            ONLY from retrieved sources, with citations + confidence
    document_simplifier.py PDF text extraction + AI simplification
  routers/
    auth.py               POST /auth/send-otp, /auth/verify-otp
    chat.py                 Chat sessions + /chat/query (the core guidance engine)
    laws.py                  Law & Case Explorer: /laws/search, /laws/{id}
    documents.py              Document Simplifier: upload + background processing
    admin.py                  Admin panel: stats, audit logs, source verification
    ingestion.py               Admin-only: add new laws/cases into the knowledge base
schema.sql              Raw SQL mirror of models.py, for manual DB review/setup
alembic/                 Migrations (auto-generated from models.py)
docker-compose.yml       Local Postgres+pgvector and the API service
```

## Why it's built this way

- **Source-grounded RAG, not free generation.** `guidance.py` never lets the LLM answer from
  its own training knowledge — it retrieves real `legal_sources` chunks first and instructs
  Claude to answer only from those, returning which sources it used and a confidence level.
  This is the "reduce hallucination" requirement from the pitch deck.
- **Admin verification gate.** New legal sources go into the DB with `is_verified=False`.
  Both the RAG retrieval (`retrieval.py`) and the public Law Explorer (`laws.py`) only
  ever surface `is_verified=True` sources — an admin has to sign off first
  (`POST /admin/legal-sources/{id}/verify`), so bad or unreviewed content never reaches users.
- **OTP phone login**, no passwords — matches the "mobile number + OTP" requirement.
- **Hybrid search** (`retrieval.py`) combines pgvector cosine similarity with Postgres
  full-text search via reciprocal rank fusion, since legal terms often need exact keyword
  matches (e.g. "Article 21") that pure embeddings can miss.
- **Audit log** (`admin_audit_logs`) records logins, chat queries, uploads, etc. so the
  admin panel (`admin.py`) has real activity data to show.

## Running locally

```bash
cp .env.example .env          # fill in JWT secret, Anthropic key, etc.
docker compose up -d db       # start Postgres+pgvector only
pip install -r requirements.txt
alembic revision --autogenerate -m "init"
alembic upgrade head
uvicorn app.main:app --reload
```

API docs then at `http://localhost:8000/docs`.

## Populating the knowledge base

Before the chat/guidance endpoint can answer anything, you need verified `legal_sources`.
As an admin (after OTP login with `ADMIN_BOOTSTRAP_PHONE`):

```bash
curl -X POST http://localhost:8000/admin/legal-sources \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
        "source_type": "constitution_article",
        "title": "Article 21 - Protection of Life and Personal Liberty",
        "citation": "Constitution of India, Art. 21",
        "full_text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        "auto_verify": true
      }'
```

This chunks + embeds the text automatically. Repeat for Constitution articles, Act sections,
landmark judgments, and NALSA resources — this seeding work (plus getting authoritative,
correctly-licensed source text) is the main remaining task before the MVP is demo-ready.

## Voice (STT/TTS)

`POST /chat/voice-query` (multipart form: `session_id`, `audio` file, optional `language`)
- Transcribes with **faster-whisper**, running locally — no API key needed, works offline,
  handles Hindi/English including code-switched speech.
- Runs the transcript through the same `generate_guidance()` used by text chat (same
  citations + confidence behavior).
- Speaks the reply back with **edge-tts** (free, no API key) and returns `reply_audio_url`,
  served from `/media/...` (mounted from `AUDIO_STORAGE_DIR`).
- Swap `app/rag/voice.py` for a paid cloud provider (Sarvam AI, Google, Azure) later if you
  want higher accuracy on Indian languages — the function signatures stay the same.

## OCR for scanned documents

`document_simplifier.py` now tries native PDF text extraction first, and if that comes back
mostly empty (a strong signal it's a scanned image, common for older judgments), falls back
to OCR via **Tesseract** with Hindi + English language packs (`hin+eng`). Requires the
`tesseract-ocr`, `tesseract-ocr-hin`, and `poppler-utils` system packages — already added to
the `Dockerfile`. If you're running outside Docker, install them yourself:
`apt install tesseract-ocr tesseract-ocr-hin poppler-utils` (or the equivalent for your OS).

## SMS / OTP delivery

`app/core/sms.py` has working integrations for **Twilio** and **MSG91**, selected via
`SMS_PROVIDER` in `.env`. This is the one piece that genuinely can't be finished by writing
more code: sending real SMS requires an account with a provider, since that's a paid,
identity-verified service (anti-fraud regulation, not a technical gap). Two options:

1. **Twilio** — sign up at twilio.com, get `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` /
   a phone number, drop them into `.env`, set `SMS_PROVIDER=twilio`. Works globally.
2. **MSG91** — sign up at msg91.com, get an `MSG91_AUTH_KEY`, set `SMS_PROVIDER=msg91`.
   Usually cheaper for Indian numbers specifically.

Until you do that, `SMS_PROVIDER=dev` (the default) logs the OTP to the console so you can
log in and test everything else end-to-end without spending money on SMS.

## Remaining production hardening (not required to run/demo the platform)

- **Rate limiting** on `/auth/send-otp` to prevent SMS-bombing abuse.
- **File storage** — documents/audio are saved to local disk; swap for S3/GCS at real scale.
- **CORS** — currently wide open (`allow_origins=["*"]`); restrict to your real frontend domain.
- **Whisper model size** — `small` balances speed/accuracy; bump to `medium` if you have GPU
  headroom and want better Hindi accuracy.
