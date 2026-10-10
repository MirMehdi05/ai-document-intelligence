
# AI Document Intelligence Platform

A local Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, search their contents, and ask natural-language questions using a locally hosted large language model.

Built with Python, FastAPI, Streamlit, Sentence Transformers, FAISS, and Ollama.

## Features

- Upload and process text-based PDF documents.
- Extract text and split documents into searchable chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and persist vectors using FAISS.
- Retrieve relevant information using hybrid semantic and keyword search.
- Generate document-grounded answers using a local Ollama LLM.
- Display retrieved source chunks for transparency.
- Search across documents or filter by a selected document.
- Prevent repeated indexing of identical newly uploaded PDFs using SHA-256 hashes.
- Preserve indexed documents after application restarts.
- Return an insufficient-information response when an answer is not supported by the supplied context.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| Retrieval | Hybrid semantic and keyword ranking |
| Local LLM | Llama 3.2 via Ollama |
| PDF extraction | pypdf |
| API testing | Swagger UI |

## Architecture

PDF Upload
    |
    v
PDF Text Extraction
    |
    v
Text Chunking
    |
    v
Sentence Transformer Embeddings
    |
    v
FAISS Vector Index + Metadata
    |
    v
Hybrid Retrieval
    |
    v
Context Construction
    |
    v
Ollama Local LLM
    |
    v
Document-Grounded Answer + Sources

## Getting Started

### Prerequisites

- Python and pip
- Ollama installed locally
- Git (for version control)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-document-intelligence.git
cd ai-document-intelligence
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Set these values in `.env`:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:latest
RAG_MIN_SCORE=0.10
```

Note: The current hybrid retrieval implementation does not use RAG_MIN_SCORE; the setting is retained for compatibility with earlier experiments.

### 5. Prepare the local language model

```powershell
ollama pull llama3.2:latest
```

Make sure the Ollama service is running.

### 6. Start the backend

```powershell
python -m uvicorn main:app --reload
```

FastAPI documentation:

http://127.0.0.1:8000/docs

### 7. Start the frontend

Open another terminal:

```powershell
streamlit run streamlit_app.py
```

Open:

http://localhost:8501

## How to Use

1. Upload a text-based PDF using the Streamlit interface.
2. Click **Index PDF**.
3. Select the indexed document.
4. Enter a question about its contents.
5. View the generated answer and retrieved source chunks.

Previously indexed documents remain available after restarting the application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/rag/upload | Upload and index a PDF |
| GET | /api/rag/documents | List indexed documents |
| POST | /api/rag/ask | Ask a document-grounded question |
| GET | /health | Backend health check |

Example question request:

```json
{
  "question": "What are the main responsibilities described in the document?",
  "top_k": 8,
  "document_id": null
}
```

## Verification

The local MVP was manually tested for:

- Successful PDF extraction and indexing.
- Accurate responses to five resume-based questions.
- Refusal to invent unsupported salary information.
- FAISS index persistence across server restarts.
- Duplicate-upload prevention for newly indexed documents.
- Successful integration between Streamlit, FastAPI, and Ollama.

These are local functional checks, not a comprehensive production evaluation.

## Limitations

- Scanned or image-only PDFs require OCR, which is not implemented.
- Answers depend on the quality of extracted text and retrieved context.
- Hybrid ranking and LLM generation can still produce imperfect results.
- Documents and indexes are stored locally.
- The application is a local prototype without authentication, user isolation, or production security controls.
- Existing indexed documents created before hash-based deduplication may not have file-hash metadata.

## Future Enhancements

- OCR for scanned documents.
- Improved document-aware chunking and retrieval reranking.
- Automated retrieval and answer-quality evaluations.
- Document deletion and index management.
- Authentication and multi-user support.
- Docker-based deployment.

## Privacy

Document contents, local FAISS indexes, and environment secrets are excluded from the public repository. The application uses a locally hosted LLM for answer generation.

## Project Status

Local MVP completed and functionally verified.
