
from app.services.llm_service import LLMService

llm = LLMService()

try:
    llm.generate("   ")
except ValueError as exc:
    print(f"Validation passed: {exc}")