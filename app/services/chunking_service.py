import re


SECTION_DEFINITIONS = {
    "employment": {
        "label": "Professional Experience",
        "aliases": {
            "experience", "professional experience", "work experience",
            "employment", "employment history", "work history",
            "professional background", "career history", "internships",
            "internship experience", "work experience and internships",
        },
    },
    "training": {
        "label": "Training & Certifications",
        "aliases": {
            "training", "courses", "coursework", "certifications",
            "certificates", "certifications and training",
            "training and certifications", "courses and certifications",
            "professional development", "workshops", "bootcamps",
            "licenses and certifications",
        },
    },
    "education": {
        "label": "Education",
        "aliases": {
            "education", "educational background", "academic background",
            "academic qualifications", "qualifications", "education and qualifications",
        },
    },
    "skills": {
        "label": "Technical Skills",
        "aliases": {
            "skills", "technical skills", "technical expertise",
            "core competencies", "technologies", "tools and technologies",
            "technical skills and tools", "skills and technologies",
        },
    },
    "projects": {
        "label": "Projects",
        "aliases": {
            "projects", "academic projects", "personal projects",
            "key projects", "selected projects", "project experience",
        },
    },
    "summary": {
        "label": "Professional Summary",
        "aliases": {
            "summary", "professional summary", "profile", "professional profile",
            "career objective", "objective", "about me",
        },
    },
    "achievements": {
        "label": "Achievements",
        "aliases": {"achievements", "awards", "honors", "honours"},
    },
    "publications": {
        "label": "Publications",
        "aliases": {"publications", "research publications", "papers"},
    },
}


def _normalize_heading(line: str) -> str:
    cleaned = re.sub(r"^[\s•▪◦\-–—*#\d.)]+", "", line.strip())
    cleaned = re.sub(r"[:|]+$", "", cleaned)
    return re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()


def detect_section_heading(line: str) -> dict | None:
    """Return a known resume section for a heading-like line, if recognized."""
    if not line or len(line.strip()) > 90:
        return None

    normalized = _normalize_heading(line)
    if not normalized:
        return None

    for section_type, definition in SECTION_DEFINITIONS.items():
        if normalized in definition["aliases"]:
            return {"section": definition["label"], "section_type": section_type}

    return None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_pages_with_sections(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk extracted PDF pages while retaining the active resume section.

    Section state carries across page boundaries, so a heading at the bottom of
    one page still labels content at the top of the next page.
    """
    chunks = []
    active_section = {"section": "General", "section_type": "other"}

    for page in pages:
        page_number = page["page_number"]
        page_runs = []
        run_lines = []

        for line in (page.get("text") or "").splitlines():
            heading = detect_section_heading(line)
            if heading:
                if run_lines:
                    page_runs.append((active_section, "\n".join(run_lines)))
                    run_lines = []
                active_section = heading
                continue

            if line.strip():
                run_lines.append(line.strip())

        if run_lines:
            page_runs.append((active_section, "\n".join(run_lines)))

        page_chunk_index = 0
        for section, run_text in page_runs:
            for text in chunk_text(run_text, chunk_size=chunk_size, overlap=overlap):
                section_prefix = (
                    f"[Document section: {section['section']}]\n"
                    if section["section_type"] != "other"
                    else ""
                )
                chunks.append({
                    "page_number": page_number,
                    "page_chunk_index": page_chunk_index,
                    "section": section["section"],
                    "section_type": section["section_type"],
                    "text": section_prefix + text,
                })
                page_chunk_index += 1

    return chunks
