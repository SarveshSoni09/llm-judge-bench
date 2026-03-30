"""Google Gemini provider for LLM-as-a-Judge evaluation.

Uses the new google-genai SDK (replaces deprecated google-generativeai).
Install: pip install google-genai
"""

import time
import os
from typing import Optional

from llm_judge_bench.providers.base import BaseProvider, LLMResponse


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models (free tier compatible).

    Uses Gemini 1.5 Flash by default, which offers 15 RPM on the free tier.
    Built-in rate-limit handling with exponential backoff.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite"
    ):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY or pass api_key=."
            )

        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._last_request_time = 0.0

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate a response with automatic rate-limit handling."""
        from google.genai import types

        self._rate_limit_wait()

        start = time.time()
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                latency = (time.time() - start) * 1000
                self._last_request_time = time.time()

                return LLMResponse(
                    text=response.text.strip(),
                    model=self._model_name,
                    latency_ms=round(latency, 2),
                )
            except Exception as e:
                err = str(e).lower()
                if (
                    "429" in str(e)
                    or "rate" in err
                    or "quota" in err
                    or "resource" in err
                ):
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    def _rate_limit_wait(self):
        """Enforce minimum gap between requests (free tier: 15 RPM)."""
        min_interval = 5.0  # slightly over 4s to stay safely under 15 RPM
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    @property
    def model_name(self) -> str:
        return self._model_name
