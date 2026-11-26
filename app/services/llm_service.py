import os
import time

from dotenv import load_dotenv

load_dotenv()


class LLMService:
    """Generate answers using local Ollama or Gemini with a free-tier fallback."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        self.client = None
        self.fallback_client = None
        self.fallback_model = None

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
            from openai import OpenAI

            self.model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
            api_key = os.getenv("LLM_API_KEY", "").strip()
            self.is_configured = bool(api_key)
            if api_key:
                base_url = os.getenv(
                    "LLM_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.client = OpenAI(
                    api_key=api_key, base_url=base_url,
                    timeout=90.0, max_retries=0,
                )
                fallback_model = os.getenv(
                    "LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite"
                ).strip()
                if fallback_model and fallback_model != self.model:
                    self.fallback_model = fallback_model
                    self.fallback_client = self.client
            return

        raise ValueError("Unsupported LLM_PROVIDER. Use 'ollama' or 'openai-compatible'.")

    def _gemini_completion(self, model: str, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if not self.is_configured or self.client is None:
            raise RuntimeError(
                "LLM is not configured. Add an API key in Streamlit secrets "
                "or use Ollama locally."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if self.provider == "ollama":
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": 0.1,
                        "num_predict": 220,
                        "num_ctx": 2048,
                        "num_thread": 4,
                    },
                    think=False,
                )
                answer = (response.message.content or "").strip()
                return answer or "I could not generate an answer. Please try again."
            except Exception as exc:
                print(f"[LLM] Ollama generation failed: {exc}", flush=True)
                raise RuntimeError(
                    "Local answer generation failed. Please try again."
                ) from exc

        last_error = None
        # Retry the selected model briefly, then use the separate Flash-Lite fallback.
        for attempt, delay in enumerate((1, 2), start=1):
            try:
                answer = self._gemini_completion(self.model, messages)
                return answer or "I could not generate an answer. Please try again."
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                if status_code not in self.RETRYABLE_STATUS_CODES:
                    break
                if attempt < 2:
                    time.sleep(delay)

        if self.fallback_model and self.fallback_client:
            try:
                response = self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                )
                answer = (response.choices[0].message.content or "").strip()
                if answer:
                    print(
                        f"[LLM] Primary model unavailable; fallback {self.fallback_model} succeeded.",
                        flush=True,
                    )
                    return answer
            except Exception as fallback_error:
                last_error = fallback_error
                print(
                    f"[LLM] Gemini fallback {self.fallback_model} failed: {fallback_error}",
                    flush=True,
                )

        status_code = getattr(last_error, "status_code", None)
        print(f"[LLM] Generation failed: {last_error}", flush=True)
        if status_code == 503:
            raise RuntimeError(
                "Gemini is temporarily overloaded. Both the primary model and "
                "the fallback are unavailable right now. Your document search "
                "still worked; see the retrieved sources below and try again shortly."
            ) from last_error
        if status_code == 429:
            raise RuntimeError(
                "The Gemini free-tier request limit may have been reached. "
                "Your document search still worked; see the retrieved sources below."
            ) from last_error
        raise RuntimeError(
            "Answer generation is temporarily unavailable. Your document search "
            "still worked; see the retrieved sources below."
        ) from last_error
