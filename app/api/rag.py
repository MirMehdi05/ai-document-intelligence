from functools import lru_cache
from pathlib import Path
import re
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from app.services.rag_service import RAGService

router = APIRouter(prefix='/api/rag', tags=['RAG'])
UPLOAD_DIR = Path('data/raw')
MAX_BYTES = 10 * 1024 * 1024

@lru_cache(maxsize=1)
def get_service():
    return RAGService()

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    document_id: str | None = None

@router.post('/upload')
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or '').name
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(400, 'Only PDF files are accepted.')
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / (uuid.uuid4().hex + '.pdf')
    try:
        with target.open('wb') as output:
            size = 0
            header = b''
            while chunk := await file.read(1024 * 1024):
                if not header: header = chunk[:5]
                size += len(chunk)
                if size > MAX_BYTES: raise HTTPException(413, 'PDF exceeds 10 MB limit.')
                output.write(chunk)
        if header != b'%PDF-':
            raise HTTPException(400, 'Invalid PDF signature.')
        try:
            return get_service().add_document(str(target), filename)
        except Exception as exc:
            raise HTTPException(400, f'Could not process PDF: {exc}') from exc
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

@router.get('/documents')
def list_documents():
    return {'documents': get_service().documents()}

@router.post('/ask')
def ask_question(request: QuestionRequest):
    try:
        return get_service().ask(request.question, request.top_k, request.document_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, 'RAG request failed. Check server logs.') from exc
