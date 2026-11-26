import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="AI Document Intelligence", page_icon="📄", layout="wide")


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


os.environ["LLM_PROVIDER"] = get_setting("LLM_PROVIDER", "openai-compatible").strip().lower()
for setting in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OLLAMA_BASE_URL", "OLLAMA_MODEL"):
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

with st.sidebar:
    st.header("Documents")
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF", type=["pdf"],
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
                with st.spinner("Extracting pages and creating embeddings..."):
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(pdf_bytes)
                        temporary_path = temp_file.name
                    result = service.add_document(temporary_path, uploaded_pdf.name)
                st.session_state["chat_messages"] = []
                status = "Already indexed" if result.get("already_indexed") else "Indexed successfully"
                st.success(f"{status}: {result['filename']} · {result['chunks']} chunks")
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
            st.write(f"**{document['filename']}**  \n{document['chunks']} chunks")
    else:
        st.caption("No documents indexed in this session yet.")
    top_k = st.slider("Retrieved passages", 1, 10, 5)
    if st.button("Clear conversation"):
        st.session_state["chat_messages"] = []
        st.rerun()

st.session_state.setdefault("chat_messages", [])
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
                result = service.ask(question.strip(), top_k=top_k, chat_history=history)
            st.session_state["chat_messages"].append({
                "role": "assistant", "content": result["answer"], "sources": result["sources"]
            })
        except Exception as exc:
            st.session_state["chat_messages"].append({
                "role": "assistant", "content": f"**Couldn't generate an answer:** {exc}", "sources": []
            })
        st.rerun()
else:
    st.info("Upload a PDF from the sidebar to start chatting with your documents.")

with st.expander("Technical architecture"):
    st.markdown(
        "- **PDF extraction:** pypdf, with page-level text tracking\n"
        "- **Chunking:** overlapping text chunks\n"
        "- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)\n"
        "- **Vector search:** FAISS with hybrid semantic and keyword ranking\n"
        "- **Answer generation:** Gemini API in the hosted demo; Ollama locally\n"
        "- **Conversation:** session-based chat history for follow-up questions"
    )

st.caption(
    "Privacy: documents are stored in temporary, isolated browser-session indexes. "
    "Use public or sample PDFs only; files may be cleared when the app restarts."
)
