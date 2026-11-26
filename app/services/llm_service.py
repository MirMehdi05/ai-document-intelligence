import os
import time

from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """Generate answers using local Ollama or a hosted OpenAI-compatible API."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if self.provider == "ollama":
            import ollama
            self.model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
            self.client = ollama.Client(
                host=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                timeout=90.0,
            )
            self.is_configured = True
            return
        if self.provider in {"openai-compatible", "openai"}:
            self.model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
            api_key = os.getenv("LLM_API_KEY", "").strip()
            self.is_configured = bool(api_key)
            self.client = None
            if api_key:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=os.getenv(
                        "LLM_BASE_URL",
                        "https://generativelanguage.googleapis.com/v1beta/openai/",
                    ),
                    timeout=90.0,
                    max_retries=0,
                )
            return
        raise ValueError("Unsupported LLM_PROVIDER. Use 'ollama' or 'openai-compatible'.")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not self.is_configured or self.client is None:
            raise RuntimeError("LLM is not configured. Add an API key in Streamlit secrets or use Ollama locally.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        attempts = 3 if self.provider != "ollama" else 1
        for attempt in range(attempts):
            try:
                if self.provider == "ollama":
                    response = self.client.chat(
                        model=self.model, messages=messages,
                        options={"temperature": 0.1, "num_predict": 220,
                                 "num_ctx": 2048, "num_thread": 4},
                        think=False,
                    )
                    answer = (response.message.content or "").strip()
                else:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=0.1,
                    )
                    answer = (response.choices[0].message.content or "").strip()
                return answer or "I could not generate an answer. Please try again."
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = status_code in {429, 500, 502, 503, 504}
                if retryable and attempt < attempts - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"[LLM] Generation failed: {exc}", flush=True)
                if status_code == 503:
                    raise RuntimeError("Gemini is temporarily busy. Please wait a moment and ask again.") from exc
                if status_code == 429:
                    raise RuntimeError("The Gemini API rate limit was reached. Please wait before retrying.") from exc
                raise RuntimeError("Answer generation failed. Please try again shortly.") from exc
        raise RuntimeError("Answer generation failed. Please try again shortly.")
