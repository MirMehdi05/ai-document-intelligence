from pathlib import Path
from pypdf import PdfReader


def extract_pages_from_pdf(file_path: str) -> list[dict]:
    """Extract text page-by-page so retrieved chunks retain page citations."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append({"page_number": page_number, "text": page_text})
    return pages


def extract_text_from_pdf(file_path: str) -> str:
    """Backward-compatible full-document text extraction."""
    return "\n".join(page["text"] for page in extract_pages_from_pdf(file_path))
