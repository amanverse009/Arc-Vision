import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db, AsyncSessionLocal
from app.models import User, UserDocument, DocumentStatus, AdminAuditLog
from app.rag.document_simplifier import extract_text_from_pdf, simplify_document_text
from app.schemas import DocumentUploadResponse, DocumentStatusResponse

router = APIRouter(prefix="/documents", tags=["document-simplifier"])

from app.config import settings

STORAGE_DIR = settings.DOCUMENT_STORAGE_DIR
ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


async def _process_document(document_id: str, language: str):
    """Background task: extract text + simplify, running in its own DB session."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(UserDocument).where(UserDocument.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return
        try:
            with open(doc.storage_path, "rb") as f:
                raw_bytes = f.read()
            extracted = extract_text_from_pdf(raw_bytes)
            if not extracted.strip():
                raise ValueError("No extractable text found (may need OCR for scanned PDFs)")

            doc.extracted_text = extracted
            doc.simplified_summary = simplify_document_text(extracted, language=language)
            doc.status = DocumentStatus.simplified
        except Exception as exc:  # noqa: BLE001 - want to record any failure reason
            doc.status = DocumentStatus.failed
            doc.simplified_summary = f"Could not process document: {exc}"
        await db.commit()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "hi",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF documents are supported currently")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    os.makedirs(STORAGE_DIR, exist_ok=True)
    stored_name = f"{uuid.uuid4()}_{file.filename}"
    storage_path = os.path.join(STORAGE_DIR, stored_name)
    with open(storage_path, "wb") as f:
        f.write(contents)

    doc = UserDocument(
        user_id=user.id,
        original_filename=file.filename,
        storage_path=storage_path,
        status=DocumentStatus.processing,
    )
    db.add(doc)
    db.add(AdminAuditLog(actor_user_id=user.id, event_type="doc_uploaded", event_meta={"filename": file.filename}))
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(_process_document, str(doc.id), language)

    return DocumentUploadResponse(document_id=doc.id, status=doc.status.value)


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(UserDocument).where(UserDocument.id == document_id, UserDocument.user_id == user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=doc.id, status=doc.status.value, simplified_summary=doc.simplified_summary
    )
