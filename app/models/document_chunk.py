from dataclasses import dataclass


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int