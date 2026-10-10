
from app.services.llm_service import LLMService

llm = LLMService()

answer = llm.generate(
    "Explain Retrieval-Augmented Generation in three simple points."
)

print("=" * 60)
print("OLLAMA PYTHON INTEGRATION TEST")
print("=" * 60)
print(answer)