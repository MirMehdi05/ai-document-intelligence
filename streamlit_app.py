import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

API = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000')
st.set_page_config(page_title='AI Document Intelligence', page_icon='📄')
st.title('📄 AI Document Intelligence')
st.caption('Upload PDFs and ask questions using local Ollama + FAISS retrieval.')

with st.sidebar:
    st.header('Documents')
    uploaded = st.file_uploader('Upload PDF', type=['pdf'])
    if st.button('Index PDF', disabled=uploaded is None):
        try:
            with st.spinner('Extracting and indexing...'):
                r = requests.post(f'{API}/api/rag/upload', files={'file': (uploaded.name, uploaded.getvalue(), 'application/pdf')}, timeout=180)
                r.raise_for_status()
                st.success(f"Indexed {r.json()['chunks']} chunks")
        except Exception as exc: st.error(str(exc))
    try:
        r = requests.get(f'{API}/api/rag/documents', timeout=30)
        r.raise_for_status()
        docs = r.json()['documents']
    except Exception as exc:
        st.error(f'API unavailable: {exc}')
        docs = []
    choices = {'All documents': None}
    choices.update({f"{d['filename']} ({d['document_id'][:8]})": d['document_id'] for d in docs})
    selection = st.selectbox('Search within', list(choices))
    top_k = st.slider('Retrieved chunks', 1, 10, 3)

question = st.text_input('Ask a question about your documents')
if st.button('Ask', type='primary', disabled=not question.strip()):
    try:
        with st.spinner('Generating answer locally...'):
            r = requests.post(f'{API}/api/rag/ask', json={'question': question, 'top_k': top_k, 'document_id': choices[selection]}, timeout=300)
            r.raise_for_status()
            result = r.json()
        st.subheader('Answer')
        st.write(result['answer'])
        with st.expander('Retrieved sources'):
            for source in result['sources']:
                st.markdown(f"**{source['source_file']} · chunk {source['chunk_index']} · similarity {source['similarity']:.3f}**")
                st.write(source['text'])
    except Exception as exc: st.error(str(exc))
