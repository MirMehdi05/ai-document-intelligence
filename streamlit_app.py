import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_BATCH = 5


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


os.environ["LLM_PROVIDER"] = get_setting("LLM_PROVIDER", "openai-compatible").strip().lower()
for setting in (
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODEL",
    "OLLAMA_BASE_URL", "OLLAMA_MODEL",
):
    value = get_setting(setting)
    if value:
        os.environ[setting] = value

from app.services.rag_service import RAGService


def get_session_service() -> RAGService:
    if "rag_service" not in st.session_state:
        session_id = st.session_state.setdefault("rag_session_id", uuid.uuid4().hex)
        storage_path = Path(tempfile.gettempdir()) / "ai-document-intelligence" / session_id
        st.session_state["rag_service"] = RAGService(storage_dir=storage_path)
    return st.session_state["rag_service"]


st.title("📄 AI Document Intelligence")
st.caption("Chat with your PDFs. Get document-grounded answers with page-level sources.")

try:
    service = get_session_service()
except Exception as exc:
    st.error(f"Could not initialise the document engine: {exc}")
    st.stop()

st.session_state.setdefault("chat_messages", [])

with st.sidebar:
    st.header("Documents")
    uploaded_pdfs = st.file_uploader(
        "Upload text-based PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to 5 PDFs per batch, 10 MB each. Scanned/image-only PDFs need OCR.",
    )
    if st.button("Index selected PDFs", type="primary", disabled=not uploaded_pdfs):
        files_to_process = uploaded_pdfs[:MAX_FILES_PER_BATCH]
        if len(uploaded_pdfs) > MAX_FILES_PER_BATCH:
            st.warning(f"Only the first {MAX_FILES_PER_BATCH} files in this batch will be processed.")
        added_count = 0
        for uploaded_pdf in files_to_process:
            temporary_path = None
            try:
                pdf_bytes = uploaded_pdf.getvalue()
                if len(pdf_bytes) > MAX_PDF_BYTES:
                    st.error(f"{uploaded_pdf.name}: exceeds the 10 MB per-file limit.")
                    continue
                if not pdf_bytes.startswith(b"%PDF-"):
                    st.error(f"{uploaded_pdf.name}: this does not appear to be a valid PDF.")
                    continue

                with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(pdf_bytes)
                        temporary_path = temp_file.name
                    result = service.add_document(temporary_path, uploaded_pdf.name)

                if result.get("already_indexed"):
                    st.info(f"Already indexed: {result['filename']}")
                else:
                    added_count += 1
                    st.success(f"Indexed: {result['filename']} · {result['chunks']} chunks")
            except Exception as exc:
                st.error(f"Could not index {uploaded_pdf.name}: {exc}")
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)

        if added_count:
            # A fresh conversation avoids carrying answers grounded in an older document set.
            st.session_state["chat_messages"] = []
            st.session_state["chat_scope"] = "all"
            st.success(f"Added {added_count} new document(s). Your conversation has been reset.")

    st.divider()
    st.subheader("Your indexed documents")
    documents = service.documents()
    if documents:
        for document in documents:
            st.write(f"**{document['filename']}**  \n{document['chunks']} chunks")

        scope_by_label = {"All documents": None}
        for document in documents:
            label = f"{document['filename']} · {document['document_id'][:8]}"
            scope_by_label[label] = document["document_id"]
        selected_label = st.selectbox("Search scope", list(scope_by_label))
        selected_document_id = scope_by_label[selected_label]
    else:
        st.caption("No documents indexed in this session yet.")
        selected_document_id = None

    top_k = st.slider("Retrieved passages", 1, 10, 5)
    if st.button("Clear conversation"):
        st.session_state["chat_messages"] = []
        st.rerun()

# Changing the search scope starts a clean conversation so earlier context cannot
# accidentally influence answers grounded in a different document set.
current_scope = selected_document_id or "all"
previous_scope = st.session_state.get("chat_scope")
if previous_scope is not None and previous_scope != current_scope:
    st.session_state["chat_messages"] = []
    st.info("Search scope changed. The conversation was cleared.")
st.session_state["chat_scope"] = current_scope

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View sources"):
                for index, source in enumerate(message["sources"], start=1):
                    page = source.get("page_number")
                    page_text = f" · page {page}" if page else ""
                    with st.expander(
                        f"[Source {index}] {source['source_file']}{page_text} · "
                        f"similarity {source['similarity']:.3f}"
                    ):
                        st.write(source["text"])

if service.documents():
    question = st.chat_input("Ask a question about your documents...")
    if question and question.strip():
        history = st.session_state["chat_messages"][-6:]
        st.session_state["chat_messages"].append({"role": "user", "content": question.strip()})
        try:
            with st.spinner("Searching your documents and generating an answer..."):
                result = service.ask(
                    question.strip(),
                    top_k=top_k,
                    document_id=selected_document_id,
                    chat_history=history,
                )
            st.session_state["chat_messages"].append({
                "role": "assistant", "content": result["answer"], "sources": result["sources"]
            })
        except Exception as exc:
            st.session_state["chat_messages"].append({
                "role": "assistant", "content": f"**Couldn't generate an answer:** {exc}", "sources": []
            })
        st.rerun()
else:
    st.info("Upload one or more PDFs from the sidebar to start chatting with your documents.")

with st.expander("Technical architecture"):
    st.markdown(
        "- **PDF extraction:** pypdf, with page-level text tracking\n"
        "- **Chunking:** overlapping text chunks\n"
        "- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)\n"
        "- **Vector search:** FAISS with hybrid semantic and keyword ranking\n"
        "- **Answer generation:** Gemini API in the hosted demo; Ollama locally\n"
        "- **Conversation:** session-based chat history, reset when the search scope changes"
    )

st.caption(
    "Privacy: documents are stored in temporary, isolated browser-session indexes. "
    "Use public or sample PDFs only; files may be cleared when the app restarts."
)
