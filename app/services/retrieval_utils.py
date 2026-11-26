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


def has_current_employment_evidence(sources: list[dict]) -> bool:
    """
    A current-role answer needs evidence from an employment section and a
    current-status marker. A course provider/date range is not sufficient.
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
