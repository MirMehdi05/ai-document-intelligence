
from app.services.rag_service import RAGService

PDF_PATH = "data/raw/sample.pdf"

rag = RAGService(PDF_PATH)

question = "What is the capital of Japan?"

result = rag.ask(question, top_k=3)

print("=" * 60)
print("AI DOCUMENT INTELLIGENCE — RAG TEST")
print("=" * 60)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(
        f"\nChunk {source['chunk_index']} "
        f"(similarity: {source['similarity']})"
    )
    print(source["text"])