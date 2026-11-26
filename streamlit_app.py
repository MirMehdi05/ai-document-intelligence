import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


APP_ENV = get_setting("APP_ENV", "development").strip().lower()
LLM_PROVIDER = get_setting(
    "LLM_PROVIDER",
    "openai-compatible" if APP_ENV == "production" else "ollama",
).strip().lower()

os.environ["LLM_PROVIDER"] = LLM_PROVIDER
for setting in (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
):
    value = get_setting(setting)
    if value:
        os.environ[setting] = value

from app.services.rag_service import RAGService


def get_session_service() -> RAGService:
    if "rag_service" not in st.session_state:
        session_id = st.session_state.setdefault("rag_session_id", uuid.uuid4().hex)
        storage_path = (
            Path(tempfile.gettempdir())
            / "ai-document-intelligence"
            / session_id
        )
        st.session_state["rag_service"] = RAGService(storage_dir=storage_path)
    return st.session_state["rag_service"]


st.title("📄 AI Document Intelligence")
st.caption(
    "Upload PDFs, retrieve relevant passages with semantic search and FAISS, "
    "and ask document-grounded questions."
)

if APP_ENV == "production":
    st.info(
        "Free portfolio demo. Each browser session has its own temporary "
        "document index. Documents may be cleared when the app restarts or "
        "your session ends. Do not upload confidential or sensitive files."
    )

if LLM_PROVIDER != "ollama" and not get_setting("LLM_API_KEY"):
    st.warning(
        "Semantic retrieval works without an API key. Add a Gemini API key "
        "in app secrets to enable AI-generated answers."
    )

try:
    service = get_session_service()
except Exception as exc:
    st.error(f"Could not initialise the document engine: {exc}")
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        help="Maximum 10 MB. Scanned/image-only PDFs need OCR.",
    )

    if st.button("Index PDF", type="primary", disabled=uploaded_pdf is None):
        pdf_bytes = uploaded_pdf.getvalue()
        if len(pdf_bytes) > 10 * 1024 * 1024:
            st.error("PDF exceeds the 10 MB upload limit.")
        elif not pdf_bytes.startswith(b"%PDF-"):
            st.error("This file does not appear to be a valid PDF.")
        else:
            temporary_path = None
            try:
                with st.spinner("Extracting text and creating embeddings..."):
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        suffix=".pdf",
                        delete=False,
                    ) as temp_file:
                        temp_file.write(pdf_bytes)
                        temporary_path = temp_file.name

                    result = service.add_document(
                        temporary_path,
                        uploaded_pdf.name,
                    )

                status = (
                    "Already indexed"
                    if result.get("already_indexed")
                    else "Indexed successfully"
                )
                st.success(
                    f"{status}: {result['filename']} · {result['chunks']} chunks"
                )
            except Exception as exc:
                st.error(f"Could not index this PDF: {exc}")
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)

    st.divider()
    st.subheader("Your indexed documents")
    documents = service.documents()
    if documents:
        for document in documents:
            st.write(f"**{document['filename']}**  
{document['chunks']} chunks")
    else:
        st.caption("No documents indexed in this session yet.")

    top_k = st.slider("Retrieved chunks", 1, 10, 3)

question = st.text_input("Ask a question about your documents")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and preparing your answer..."):
            result = service.ask(question, top_k=top_k)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved evidence")
        for index, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"{index}. {source['source_file']} · chunk "
                f"{source['chunk_index']} · similarity {source['similarity']:.3f}",
                expanded=index == 1,
            ):
                st.write(source["text"])
    except Exception as exc:
        st.error(f"Could not answer this question: {exc}")

with st.expander("About this demo"):
    st.markdown(
        "- PDF extraction: pypdf\n"
        "- Chunking: overlapping text chunks\n"
        "- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)\n"
        "- Vector search: FAISS with hybrid semantic and keyword ranking\n"
        "- Answer generation: local Ollama or Gemini API\n"
        "- Privacy: isolated temporary indexes per browser session"
    )
