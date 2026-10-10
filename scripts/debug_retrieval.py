
import os
import numpy as np
import faiss

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from app.services.rag_service import RAGService
from app.services.context_service import build_context


# ==========================================
# 1. INITIALIZE RAG SERVICE
# ==========================================

rag = RAGService()

question = (
    "Which company does Mir Mohammed Mehdi "
    "currently work for?"
)

min_score = float(os.getenv("RAG_MIN_SCORE", "0.10"))

print("\n========== CONFIGURATION ==========")
print("Question:", question)
print("Similarity threshold:", min_score)


# ==========================================
# 2. GENERATE QUERY EMBEDDING
# ==========================================

query_embedding = rag.embedding_service.generate_embedding(
    question
)

query_vector = np.asarray(
    [query_embedding],
    dtype="float32"
)

faiss.normalize_L2(query_vector)


# ==========================================
# 3. SEARCH FAISS INDEX
# ==========================================

print("\n========== RETRIEVAL DIAGNOSTIC ==========")
print("Total indexed chunks:", rag.index.ntotal)

if rag.index.ntotal == 0:
    print("ERROR: No indexed documents found.")
    raise SystemExit(1)

scores, indices = rag.index.search(
    query_vector,
    min(10, rag.index.ntotal)
)

selected = []

for rank, (score, index) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    if index < 0:
        continue

    record = rag.records[int(index)]
    chunk_text = record["text"]
    similarity = float(score)

    contains_people_inc = (
        "people inc" in chunk_text.lower()
    )

    passed_threshold = similarity >= min_score

    print(f"\n========== RESULT {rank} ==========")
    print("Similarity:", round(similarity, 4))
    print("Source:", record["source_file"])
    print("Chunk index:", record["chunk_index"])
    print("Contains People Inc:", contains_people_inc)
    print("Passed threshold:", passed_threshold)
    print("Text preview:")
    print(chunk_text[:500])

    if passed_threshold:
        selected.append(chunk_text)


# ==========================================
# 4. BUILD CONTEXT
# ==========================================

context = build_context(selected)

print("\n========== CONTEXT DIAGNOSTIC ==========")
print("Selected chunks:", len(selected))
print("Context length:", len(context))
print(
    "Context contains People Inc:",
    "people inc" in context.lower()
)


# ==========================================
# 5. FINAL DIAGNOSTIC RESULT
# ==========================================

print("\n========== FINAL RESULT ==========")

if "people inc" in context.lower():
    print("SUCCESS: People Inc is present in the context.")
    print("The retrieval and context-building check passed.")
else:
    print("FAILED: People Inc is missing from the context.")
    print("Check similarity threshold and retrieval ranking.")

print("\nDiagnostic completed.")
