from dataclasses import dataclass


@dataclass
class DocumentCollection:
    document_id: str
    source_file: str
    chunks: list
    embeddings: list