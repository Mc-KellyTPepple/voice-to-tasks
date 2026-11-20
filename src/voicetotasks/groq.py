"""Minimal Groq client for speech-to-text and JSON chat, written for tight free-tier limits.

It retries 429/5xx a few times (honouring Retry-After) and turns everything else into GroqError with a
short, user-safe message. It never logs audio or transcripts.
"""
from __future__ import annotations

import time

import requests

BASE = "https://api.groq.com/openai/v1"
STT_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]
TEXT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


class GroqError(RuntimeError):
    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind   # "quota" | "unavailable" | "error"


class Groq:
    def __init__(self, api_key: str, post=None, sleep=time.sleep, retries: int = 2, timeout: int = 60):
        if not api_key:
            raise GroqError("GROQ_API_KEY is not set", "unavailable")
        self.key, self.sleep, self.retries, self.timeout = api_key, sleep, retries, timeout
        self._post = post or requests.post

    def _request(self, path: str, **kw):
        headers = {"Authorization": f"Bearer {self.key}"}
        for attempt in range(self.retries + 1):
            try:
                r = self._post(f"{BASE}{path}", headers=headers, timeout=self.timeout, **kw)
            except requests.RequestException:
                if attempt == self.retries:
                    raise GroqError("Could not reach the speech provider.", "unavailable") from None
                self.sleep(1 + attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.retries:
                    kind = "quota" if r.status_code == 429 else "unavailable"
                    raise GroqError("The speech provider is busy or out of quota right now.", kind)
                try:
                    wait = min(float(r.headers.get("retry-after", 2)), 8)
                except ValueError:
                    wait = 2
                self.sleep(wait)
                continue
            raise GroqError(f"The provider rejected the request ({r.status_code}).", "error")
        raise GroqError("unreachable", "error")   # pragma: no cover

    def transcribe(self, audio: bytes, filename: str, mime: str, model: str = STT_MODELS[0], language: str | None = "en") -> dict:
        data = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language
        return self._request("/audio/transcriptions", files={"file": (filename, audio, mime)}, data=data)

    def chat_json(self, messages: list[dict], models: list[str] | None = None, max_tokens: int = 1200) -> tuple[dict | None, str, dict]:
        """Returns (parsed JSON or None, model used, usage). Tries each model in turn."""
        import json
        last: GroqError | None = None
        for model in models or TEXT_MODELS:
            body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"}}
            if model.startswith("openai/gpt-oss"):
                body["reasoning_effort"] = "low"
            try:
                out = self._request("/chat/completions", json=body)
            except GroqError as exc:
                last = exc
                continue
            text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            try:
                return json.loads(text), model, out.get("usage") or {}
            except ValueError:
                last = GroqError("The model did not return valid JSON.", "error")
        raise last or GroqError("No model available.", "unavailable")
