import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import auth, chat, laws, documents, admin, ingestion

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Powered Legal & Constitutional Rights Guidance Platform — "
        "From Problem to Rights, not just Question to Answer."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend domain(s) before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.AUDIO_STORAGE_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.AUDIO_STORAGE_DIR), name="media")

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(laws.router)
app.include_router(documents.router)
app.include_router(admin.router)
app.include_router(ingestion.router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}


@app.get("/about")
async def about():
    return {
        "app": settings.APP_NAME,
        "description": "AI-powered legal information and guidance tool — not a substitute for a qualified lawyer.",
        "developed_by": "Aman",
    }
