import re


def classify_query_intent(question: str) -> str:
    """Classify common document questions so retrieval can respect section context."""
    text = re.sub(r"\s+", " ", question.lower()).strip()

    current_role_patterns = (
        r"\b(current|currently|present|right now)\b.{0,45}\b(role|position|job|title|work|working|employer|company|occupation|do)\b",
        r"\b(where|who)\b.{0,35}\b(work|working|employed|employer)\b",
        r"\bwhat\s+does\s+.+?\s+(?:currently\s+)?do\b",
        r"\bcurrent\s+(?:role|position|job|title|employer|company|employment)\b",
    )
    if any(re.search(pattern, text) for pattern in current_role_patterns):
        return "current_role"

    if re.search(
        r"\b(employer|employment|work experience|professional experience|"
        r"job history|work history|previous companies|companies worked|"
        r"internship experience)\b",
        text,
    ):
        return "employment"
    if re.search(r"\b(skill|skills|technology|technologies|framework|tools|tech stack)\b", text):
        return "skills"
    if re.search(r"\b(project|projects|built|developed|implemented)\b", text):
        return "projects"
    if re.search(r"\b(education|degree|college|university|qualification)\b", text):
        return "education"
    if re.search(r"\b(course|courses|training|certification|certifications|certificate)\b", text):
        return "training"
    return "general"


def expand_retrieval_query(question: str) -> str:
    """Add intent-specific concepts to the embedding query without changing the user prompt."""
    intent = classify_query_intent(question)
    expansions = {
        "current_role": (
            "current job title role position employer company employment "
            "professional experience work experience currently employed as"
        ),
        "employment": (
            "employment history professional experience work experience "
            "employer job title responsibilities organization"
        ),
        "skills": "technical skills technologies tools frameworks programming languages expertise",
        "projects": "projects portfolio developed built implemented applications solutions",
        "education": "education academic qualification degree university college",
        "training": "courses training certifications certificates professional development",
    }
    expansion = expansions.get(intent)
    return question if not expansion else f"{question} {expansion}"


def section_relevance_bonus(intent: str, section_type: str) -> float:
    """Small, transparent ranking adjustment based on the document's section label."""
    bonuses = {
        "current_role": {
            "employment": 0.35, "summary": 0.02, "training": -0.30,
            "education": -0.18, "skills": -0.10, "projects": -0.10,
        },
        "employment": {"employment": 0.30, "training": -0.20, "education": -0.10},
        "skills": {"skills": 0.25},
        "projects": {"projects": 0.25},
        "education": {"education": 0.25, "training": 0.05},
        "training": {"training": 0.25, "education": 0.05},
    }
    return bonuses.get(intent, {}).get(section_type, 0.0)


def expand_with_adjacent_employment_chunks(
    ranked: list[dict],
    top_k: int,
    neighbor_radius: int = 2,
) -> list[dict]:
    """
    Add nearby chunks around the strongest employment match.

    Resume entry titles and date ranges often land in one chunk while the
    responsibilities are in the next. Neighbor expansion restores this local
    context without crossing document or detected-section boundaries.
    """
    if not ranked:
        return []

    employment = [
        item for item in ranked
        if item.get("section_type") == "employment"
    ]
    if not employment:
        return ranked[:top_k]

    lookup = {
        (item.get("document_id"), item.get("chunk_index")): item
        for item in ranked
    }
    seed = employment[0]
    selected = []
    selected_keys = set()
    document_id = seed.get("document_id")
    chunk_index = seed.get("chunk_index")

    for offset in range(-neighbor_radius, neighbor_radius + 1):
        key = (document_id, chunk_index + offset)
        neighbor = lookup.get(key)
        if not neighbor or neighbor.get("section_type") != "employment":
            continue
        if neighbor.get("document_id") != document_id or key in selected_keys:
            continue
        enriched = dict(neighbor)
        enriched["context_expanded"] = offset != 0
        selected.append(enriched)
        selected_keys.add(key)

    # Keep enough context to bridge an entry heading/date to its responsibilities,
    # even when the UI top-k slider is set to one.
    target_count = max(top_k, min(3, len(lookup)))
    for item in ranked:
        if len(selected) >= target_count:
            break
        # Training and certification providers must not be reintroduced into
        # a current-employment context window as competing employer evidence.
        if item.get("section_type") == "training":
            continue
        key = (item.get("document_id"), item.get("chunk_index"))
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= target_count:
            break

    return selected


def has_current_employment_evidence(sources: list[dict]) -> bool:
    """
    A current-role answer needs an experience-section passage containing an
    explicit current marker. Adjacent chunks are supplied by the selector above.
    """
    current_marker = re.compile(
        r"\b(present|currently|current|ongoing|to date|till date|now)\b",
        re.IGNORECASE,
    )
    for source in sources:
        if source.get("section_type") == "employment" and current_marker.search(
            source.get("text", "")
        ):
            return True
    return False
