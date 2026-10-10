import pytest
from app.services.chunking_service import chunk_text
from app.services.context_service import build_context
from app.services.prompt_service import build_rag_prompt

def test_empty_chunking():
    assert chunk_text('') == []

def test_chunking_overlap():
    assert chunk_text('abcdefghij', chunk_size=6, overlap=2) == ['abcdef', 'efghij', 'ij']

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text('abc', chunk_size=4, overlap=4)

def test_prompt_contains_context():
    assert 'Evidence' in build_rag_prompt('What?', 'Evidence')

def test_context():
    assert 'Hello' in build_context(['Hello'])
