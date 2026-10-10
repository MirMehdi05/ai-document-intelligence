from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.search import router as search_router
from app.api.rag import router as rag_router

app = FastAPI(title='AI Document Intelligence Platform', version='1.0.0')

@app.get('/')
def home():
    return {'message': 'AI Document Intelligence Platform is running', 'status': 'success'}

@app.get('/health')
def health():
    return {'status': 'ok'}

app.include_router(search_router, prefix='/api')
app.include_router(rag_router)
