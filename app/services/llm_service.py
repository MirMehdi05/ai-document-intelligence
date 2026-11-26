import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama",
).strip().lower()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest",
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.groq.com/openai/v1",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("GROQ_API_KEY", ""),
)


class LLMService:
    """Generate answers with local Ollama or an OpenAI-compatible API."""

    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "ollama":
            import ollama

            self.client = ollama.Client(
                host=OLLAMA_BASE_URL,
                timeout=90.0,
            )
            self.model = OLLAMA_MODEL

        elif self.provider in {"openai-compatible", "openai", "groq"}:
            if not LLM_API_KEY:
                raise RuntimeError(
                    "The hosted LLM is not configured. Set LLM_API_KEY "
                    "and LLM_PROVIDER=openai-compatible in the deployment secrets."
                )

            from openai import OpenAI

            self.client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                timeout=90.0,
            )
            self.model = LLM_MODEL

        else:
            raise ValueError(
                "Unsupported LLM_PROVIDER. Use 'ollama' or 'openai-compatible'."
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        print(
            f"[LLM] Starting generation with {self.provider}:{self.model}",
            flush=True,
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            if self.provider == "ollama":
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": 0.1,
                        "num_predict": 150,
                        "num_ctx": 2048,
                        "num_thread": 4,
                    },
                    think=False,
                )
                answer = (response.message.content or "").strip()

            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_completion_tokens=256,
                )
                answer = (
                    response.choices[0].message.content or ""
                ).strip()

            print("[LLM] Generation completed.", flush=True)
            return answer or "I could not generate an answer. Please try again."

        except Exception as exc:
            print(f"[LLM] Generation failed: {exc}", flush=True)
            raise RuntimeError(
                f"{self.provider} generation failed: {exc}"
            ) from exc
