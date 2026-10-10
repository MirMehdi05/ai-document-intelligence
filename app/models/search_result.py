from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    score: float