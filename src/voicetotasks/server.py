"""Small HTTP service (standard library only).

  GET  /health      liveness
  POST /transcribe  body = raw audio (webm, ogg, mp4, wav, mp3, flac), up to ~30 s -> transcript, summary, tasks, timings
  POST /tasks       {"text": "..."} -> summary, tasks (for visitors without a microphone)

Runs on a small host with a rate-limited key: per-IP and daily limits, a size limit, audio kept in memory only,
nothing logged. Without GROQ_API_KEY it starts anyway and answers 503 on the two POST routes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .groq import STT_MODELS, Groq, GroqError
from .tasks import extract_tasks

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
MAX_AUDIO_BYTES = 1_500_000
MAX_TEXT_CHARS = 4000
MIN_WORDS = 4


class Limits:
    def __init__(self, per_ip_per_hour: int = 6, daily_cap: int = 100, clock=time.time):
        self.per_ip, self.daily_cap, self.clock = per_ip_per_hour, daily_cap, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.count = "", 0
        self.lock = threading.Lock()

    def allow(self, ip: str) -> str | None:
        with self.lock:
            now = self.clock()
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.daily_cap:
                return "The daily quota for this demo is used up. Please try again tomorrow."
            q = self.hits[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_ip:
                return f"Rate limit: at most {self.per_ip} recordings per hour per visitor."
            q.append(now)
            self.count += 1
            return None


def sniff_audio(data: bytes) -> tuple[str, str] | None:
    """(filename, mime) from the file's first bytes, or None if it is not audio we accept."""
    h = data[:12]
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return "clip.webm", "audio/webm"
    if h[:4] == b"OggS":
        return "clip.ogg", "audio/ogg"
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "clip.wav", "audio/wav"
    if h[4:8] == b"ftyp":
        return "clip.m4a", "audio/mp4"
    if h[:4] == b"fLaC":
        return "clip.flac", "audio/flac"
    if h[:3] == b"ID3" or (len(h) > 1 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0):
        return "clip.mp3", "audio/mpeg"
    return None


def speech_found(verbose: dict) -> bool:
    """Whisper invents text for silence. Treat the clip as empty when every segment is probably not speech."""
    segs = verbose.get("segments") or []
    return not segs or any((s.get("no_speech_prob") or 0) < 0.6 for s in segs)


class App:
    def __init__(self, client: Groq | None, limits: Limits | None = None):
        self.client = client
        self.limits = limits or Limits()
        self.gate = threading.BoundedSemaphore(3)   # a few at once; the provider's per-minute limit is shared

    def _unconfigured(self):
        return 503, {"error": "This demo is not configured yet (no model key on the server)."}

    def _fail(self, exc: GroqError):
        msg = ("The speech provider is busy or out of quota right now. Try again later."
               if exc.kind == "quota" else "The speech provider returned an error. Try again.")
        return 503, {"error": msg}

    def transcribe(self, audio: bytes, ip: str) -> tuple[int, dict]:
        if self.client is None:
            return self._unconfigured()
        if not audio:
            return 400, {"error": "No audio received."}
        if len(audio) > MAX_AUDIO_BYTES:
            return 413, {"error": f"Recording too large (max {MAX_AUDIO_BYTES / 1e6:.1f} MB, about 30 seconds)."}
        kind = sniff_audio(audio)
        if not kind:
            return 400, {"error": "That does not look like an audio file."}
        refused = self.limits.allow(ip)
        if refused:
            return 429, {"error": refused}
        t0 = time.monotonic()
        try:
            with self.gate:
                verbose = self.client.transcribe(audio, kind[0], kind[1], STT_MODELS[0])
                t1 = time.monotonic()
                text = (verbose.get("text") or "").strip()
                if not speech_found(verbose) or len(text.split()) < MIN_WORDS:
                    return 200, {"transcript": text if speech_found(verbose) else "", "empty": True, "tasks": [], "summary": "",
                                 "note": "I could not hear enough speech. Try again closer to the microphone.",
                                 "seconds": {"transcribe": round(t1 - t0, 1), "tasks": 0, "total": round(t1 - t0, 1)}}
                result = extract_tasks(self.client, text)
        except GroqError as exc:
            return self._fail(exc)
        except Exception as exc:   # never leak internals
            print(f"transcribe failed: {type(exc).__name__}", flush=True)
            return 502, {"error": "Something went wrong. Try again."}
        t2 = time.monotonic()
        return 200, {
            "transcript": text, "empty": False, **result, "speech_model": STT_MODELS[0],
            "audio_seconds": round(verbose.get("duration") or 0, 1),
            "seconds": {"transcribe": round(t1 - t0, 1), "tasks": round(t2 - t1, 1), "total": round(t2 - t0, 1)},
        }

    def tasks_from_text(self, text, ip: str) -> tuple[int, dict]:
        if self.client is None:
            return self._unconfigured()
        if not isinstance(text, str) or len(text.split()) < MIN_WORDS:
            return 400, {"error": "Type or paste at least a sentence."}
        if len(text) > MAX_TEXT_CHARS:
            return 400, {"error": f"Text too long (max {MAX_TEXT_CHARS} characters)."}
        refused = self.limits.allow(ip)
        if refused:
            return 429, {"error": refused}
        t0 = time.monotonic()
        try:
            with self.gate:
                result = extract_tasks(self.client, text.strip())
        except GroqError as exc:
            return self._fail(exc)
        except Exception as exc:
            print(f"tasks failed: {type(exc).__name__}", flush=True)
            return 502, {"error": "Something went wrong. Try again."}
        return 200, {"transcript": text.strip(), "empty": False, **result,
                     "seconds": {"transcribe": 0, "tasks": round(time.monotonic() - t0, 1), "total": round(time.monotonic() - t0, 1)}}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            self._json(200, {"ok": True}) if self.path == "/health" else self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path not in ("/transcribe", "/tasks"):
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "bad request"})
            if n > MAX_AUDIO_BYTES + 10_000:
                return self._json(413, {"error": "request too large"})
            body = self.rfile.read(n)
            fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            ip = fwd or self.client_address[0]
            if self.path == "/transcribe":
                code, obj = app.transcribe(body, ip)
            else:
                try:
                    data = json.loads(body or b"{}")
                    text = data.get("text") if isinstance(data, dict) else None
                except ValueError:
                    text = None
                code, obj = app.tasks_from_text(text, ip)
            self._json(code, obj)

        def log_message(self, *a):
            pass

    return Handler


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def build_app(per_ip: int = 6, daily_cap: int = 100) -> App:
    try:
        client = Groq(os.environ.get("GROQ_API_KEY", ""))
    except GroqError:
        print("GROQ_API_KEY is not set: serving /health, but the POST routes answer 503 until it is added.", flush=True)
        client = None
    return App(client, Limits(per_ip, daily_cap))


def serve(app: App, host: str = "0.0.0.0", port: int | None = None) -> None:
    port = port or int(os.environ.get("PORT", "7860"))
    print(f"listening on {host}:{port}", flush=True)
    Server((host, port), make_handler(app)).serve_forever()
