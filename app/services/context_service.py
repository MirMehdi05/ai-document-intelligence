
def build_context(
    chunks: list[str],
    max_chars: int = 6000,
) -> str:
    context_parts = []
    total_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        part = f"[Source chunk {index}]\n{chunk}\n"

        if total_chars + len(part) > max_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n".join(context_parts)