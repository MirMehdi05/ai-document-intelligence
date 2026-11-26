# AI Document Intelligence Platform

A Retrieval-Augmented Generation (RAG) application for uploading text-based PDFs, searching document contents, and asking document-grounded questions.

Built with Python, Streamlit, Sentence Transformers, FAISS, pypdf, and Ollama or Gemini API.

## Features

- Upload up to 5 text-based PDFs per batch (10 MB per file).
- Search across all indexed PDFs or limit retrieval to one selected document.
- Extract text and split it into overlapping chunks with page metadata.
- Preserve resume section headings (Experience, Training, Education, Skills, Projects, and more) in chunk metadata, including across PDF page boundaries.
- Generate semantic embeddings with Sentence Transformers.
- Store and search vectors using FAISS.
- Retrieve relevant passages using hybrid semantic, keyword, and section-aware ranking.
- Prioritize work-experience evidence for current-role questions and down-rank training/course sections.
- Prevent a training provider or certification issuer from being presented as an employer solely because it appears beside a date range.
- Require an experience-section source and a clear current-status marker before answering a current-role question.
- Generate grounded, concise answers using local Ollama or a hosted Gemini API.
- Keep answers natural and direct, without repetitive document-introduction phrases.
- Display supporting passages separately under each answer, including filename, PDF page, section, and relevance score.
- Re-index previously uploaded PDFs with the latest section-aware parser without removing other indexed documents.
- Give search-scope-aware guidance when the selected document(s) do not contain an answer.
- Support conversational follow-up questions with short session-based chat history.
- Reset conversation context when new documents are added or the search scope changes.
- Retry temporary Gemini service errors and show clearer rate-limit or availability messages.
- Prevent duplicate indexing of identical PDFs using SHA-256 hashes.
- Keep each browser session's documents in a separate temporary index.

## Architecture

\`\`\`mermaid
flowchart TD
    A["Streamlit UI"] --> B["PDF extraction with page metadata"]
    B --> C["Section-aware chunking"]
    C --> D["Sentence Transformer embeddings"]
    D --> E["FAISS vector index"]
    E --> F["Hybrid + section-aware retrieval"]
    F --> G["Context assembly with document, page, and section labels"]
    G --> H["Evidence checks for current employment"]
    H --> I["Gemini API or local Ollama"]
    I --> J["Natural answer + separate source evidence panel"]
\`\`\`

## Run locally

Requirements: Python 3.11+, Git, and Ollama for local inference.

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies with \`python -m pip install -r requirements.txt\`.
4. Copy \`.env.example\` to \`.env\`.
5. Run \`ollama pull llama3.2:latest\`.
6. Start the app with \`streamlit run streamlit_app.py\`.

By default, local development uses Ollama at http://127.0.0.1:11434. The FastAPI API in main.py remains available for local API development, but the portfolio demo runs the RAG service directly inside Streamlit.

## Free public demo deployment

The simplest no-cost hosting option for this Streamlit project is Streamlit Community Cloud. It connects directly to GitHub and deploys the app from Streamlit.

1. Push or merge the deployment branch to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select Create app.
4. Choose this repository, the branch, and \`streamlit_app.py\` as the app file.
5. Select Python 3.11 in the app's advanced settings if prompted.
6. Deploy. The app receives a public streamlit.app URL.

### Optional: enable free-tier Gemini answer generation

The hosted app works without any secrets: semantic retrieval runs and the app displays the retrieved evidence. To also generate answers:

1. Create an API key in Google AI Studio: https://aistudio.google.com/apikey.
2. Use a model enabled for your account's free tier.
3. In Streamlit Community Cloud, open App settings → Secrets and add:

    LLM_API_KEY = "your-gemini-api-key"
    LLM_MODEL = "gemini-3.6-flash"

The app defaults to the Gemini-compatible provider in the cloud. Local development continues to use Ollama from your .env file.

Google's Gemini API currently offers a free tier for selected models, subject to rate limits and model availability. Free-tier content may be used to improve Google's products, so do not upload confidential, personal, or sensitive documents. Do not enable a paid tier for this portfolio app.

## Free-hosting limitations

- Streamlit Community Cloud is free, but its compute limits are shared and can vary. The Sentence Transformer and PyTorch dependencies may take time to install and load; if the app reaches its resource limit, reduce the model workload or use a local demo recording.
- Hosted demo storage is temporary. Documents and FAISS indexes can disappear after app restart or session termination.
- Each browser session gets a separate document index. This is a portfolio demo, not a production document-management service.
- Up to 5 PDFs can be selected in one batch; each file is limited to 10 MB. Larger workloads may exceed the free app's memory or compute limits.
- The first question after deployment may take longer while the embedding model downloads and loads.
- Scanned/image-only PDFs require OCR, which is not implemented.
- Section detection recognizes common resume headings. Unrecognized or poorly extracted headings remain in the General section; verify the evidence panel for unusual layouts.
- Retrieval and generated answers can still be imperfect. The current-role evidence check is intentionally conservative and may decline to answer when a resume does not clearly mark a current job.

## API endpoints for local backend development

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Backend health check |
| POST | /api/rag/upload | Upload and index a PDF |
| GET | /api/rag/documents | List indexed documents |
| POST | /api/rag/ask | Ask a document-grounded question |

## Verification

Run the dependency-light unit tests with:

\`\`\`bash
python -m unittest discover -s tests -v
\`\`\`

After deploying the update, test with anonymized sample resumes and verify:

1. Upload a resume that has distinct Professional Experience and Training & Certifications sections.
2. Ask "What is the candidate's current role?" and confirm the answer comes from Professional Experience, not a course provider.
3. If no current position is explicitly marked, confirm the app says it cannot verify a current role instead of guessing.
4. Ask about skills, projects, and education; confirm section-aware retrieval still returns the relevant content.
5. Expand Sources & evidence and confirm each passage shows the correct filename, page, and detected section.
6. For PDFs uploaded before this update in the same browser session, select **Re-index selected PDFs** before indexing them again. This rebuilds their metadata with the new parser.

A successful GitHub merge does not itself prove a successful Streamlit cloud deployment; verify by uploading anonymized sample PDFs and asking questions after the app updates.

## Privacy

Use sample or public documents only in the hosted portfolio demo. Do not upload confidential, regulated, or personally sensitive material.
