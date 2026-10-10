
import os

import ollama
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest",
)


class LLMService:
    def __init__(self):
        self.client = ollama.Client(
            host=OLLAMA_BASE_URL,
            timeout=90.0,
        )
        self.model = OLLAMA_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        print(f"[LLM] Starting generation with {self.model}", flush=True)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 150,
                    "num_ctx": 2048,
                    "num_thread": 4,
                },
                think=False,
            )

            answer = (response.message.content or "").strip()

            print("[LLM] Generation completed.", flush=True)

            return answer or (
                "I could not generate an answer. Please try again."
            )

        except Exception as exc:
            print(f"[LLM] Generation failed: {exc}", flush=True)
            raise RuntimeError(
                f"Ollama generation failed: {exc}"
            ) from exc
