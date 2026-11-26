import hmac
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("RAG_API_TOKEN", "")
API_HEADERS = {"X-API-Key": API_TOKEN} if API_TOKEN else {}

if APP_ENV == "production" and not API_TOKEN:
    st.error("The backend API token is not configured. Add RAG_API_TOKEN to the Streamlit app secrets.")
    st.stop()


def require_access() -> None:
    """Require a shared password in production; keep local development simple."""
    if APP_ENV == "production" and not APP_PASSWORD:
        st.error(
            "The demo is locked because APP_PASSWORD has not been configured. "
            "Set APP_PASSWORD in your hosting provider's secret settings."
        )
        st.stop()

    if not APP_PASSWORD:
        return

    if st.session_state.get("authenticated", False):
        return

    st.title("🔐 AI Document Intelligence")
    st.caption("Enter the demo password to continue.")

    with st.form("access_form"):
        entered_password = st.text_input(
            "Demo password",
            type="password",
        )
        submitted = st.form_submit_button(
            "Unlock application",
            type="primary",
        )

    if submitted:
        if hmac.compare_digest(entered_password, APP_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


require_access()

st.title("📄 AI Document Intelligence")
st.caption(
    "Upload PDFs, retrieve relevant passages, and ask grounded questions "
    "with semantic search, FAISS, and an LLM."
)

if APP_ENV == "production":
    st.info(
        "Portfolio demo: documents are shared by people who have the demo "
        "password. Do not upload confidential or personal documents. "
        "The hosted demo's files and index may be cleared when the Space restarts."
    )

if LLM_PROVIDER != "ollama" and not (
    os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
):
    st.warning(
        "The hosted LLM API key is not configured. Add LLM_API_KEY in the "
        "hosting provider's secrets before asking questions."
    )

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if st.button("Index PDF", disabled=uploaded is None):
        try:
            with st.spinner("Extracting text and building embeddings..."):
                response = requests.post(
                    f"{API}/api/rag/upload",
                    headers=API_HEADERS,
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
            status = "Already indexed" if result.get("already_indexed") else "Indexed"
            st.success(
                f"{status}: {result['filename']} · {result['chunks']} chunks"
            )
        except requests.RequestException as exc:
            st.error(f"Could not index PDF: {exc}")
        except (KeyError, ValueError) as exc:
            st.error(f"Unexpected indexing response: {exc}")

    st.divider()
    st.subheader("Indexed documents")

    try:
        response = requests.get(f"{API}/api/rag/documents", headers=API_HEADERS, timeout=30)
        response.raise_for_status()
        documents = response.json().get("documents", [])
    except requests.RequestException as exc:
        st.error(f"Backend API unavailable: {exc}")
        documents = []

    choices = {"All documents": None}
    choices.update(
        {
            f"{document['filename']} ({document['document_id'][:8]})":
            document["document_id"]
            for document in documents
        }
    )

    selection = st.selectbox("Search within", list(choices))
    top_k = st.slider("Retrieved chunks", 1, 10, 3)

    if APP_PASSWORD and st.button("Lock application"):
        st.session_state["authenticated"] = False
        st.rerun()

question = st.text_input("Ask a question about your documents")

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving context and generating an answer..."):
            response = requests.post(
                f"{API}/api/rag/ask",
                headers=API_HEADERS,
                json={
                    "question": question,
                    "top_k": top_k,
                    "document_id": choices[selection],
                },
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("Retrieved sources"):
            for source in result.get("sources", []):
                st.markdown(
                    f"**{source['source_file']} · chunk "
                    f"{source['chunk_index']} · similarity "
                    f"{source['similarity']:.3f}**"
                )
                st.write(source["text"])

    except requests.RequestException as exc:
        st.error(f"Could not generate an answer: {exc}")
    except (KeyError, ValueError) as exc:
        st.error(f"Unexpected answer response: {exc}")
