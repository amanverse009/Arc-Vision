import io
import logging

import anthropic
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

from app.config import settings

logger = logging.getLogger("kanoon_wala.document_simplifier")

SIMPLIFY_SYSTEM_PROMPT = """You simplify Indian legal documents and judgments for ordinary citizens.
Explain in plain, simple {language} what the document says: who is involved, what happened,
what was decided/required, and what the reader may need to do next (if anything).
Do not invent facts not present in the document. If a section is unclear or illegible, say so.
End with a short line reminding the reader this is a simplification, not legal advice."""

# Tesseract language codes: "hin" = Hindi, "eng" = English. Both loaded so mixed
# Hindi/English judgments (common in Indian courts) OCR reasonably well.
OCR_LANGUAGES = "hin+eng"


def _extract_text_native(file_bytes: bytes) -> str:
    """Fast path: works for PDFs that have a real text layer (not scanned images)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_via_ocr(file_bytes: bytes) -> str:
    """Fallback for scanned/image-only PDFs: rasterize each page, then OCR it."""
    pages = convert_from_bytes(file_bytes, dpi=300)
    texts = []
    for i, page_image in enumerate(pages):
        try:
            texts.append(pytesseract.image_to_string(page_image, lang=OCR_LANGUAGES))
        except Exception:
            logger.exception(f"OCR failed on page {i}")
    return "\n".join(texts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Tries the fast native-text extraction first (cheap, works for born-digital PDFs).
    If that yields little/no text, falls back to OCR — handles scanned judgments,
    which are common with older Indian court documents.
    """
    native_text = _extract_text_native(file_bytes)
    if len(native_text.strip()) >= 50:  # heuristic: a real text layer will have more than this
        return native_text

    logger.info("Native text extraction came up mostly empty; falling back to OCR")
    ocr_text = _extract_text_via_ocr(file_bytes)
    return ocr_text if ocr_text.strip() else native_text


def simplify_document_text(raw_text: str, language: str = "hi") -> str:
    lang_name = "Hindi" if language == "hi" else "English"
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    # Truncate very long documents to stay within a reasonable context budget for MVP;
    # a production version should chunk + map-reduce summarize instead.
    truncated = raw_text[:15000]

    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=1200,
        system=SIMPLIFY_SYSTEM_PROMPT.format(language=lang_name),
        messages=[{"role": "user", "content": f"DOCUMENT TEXT:\n{truncated}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
