import io
import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from voicetotasks import bench
from voicetotasks.groq import Groq, GroqError
from voicetotasks.server import App, Limits, Server, make_handler, segment_info, sniff_audio, speech_found
from voicetotasks.tasks import clean
from voicetotasks.wer import bootstrap_ci, edit_distance, normalize, utterance_errors

WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 64


class FakeResp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._b, self.headers = code, body or {}, headers or {}

    def json(self):
        return self._b


class FakeClient:
    def __init__(self, text="remind me to call Sam tomorrow and send the invoice by Friday", segs=None, tasks=None):
        self.text, self.segs = text, segs
        self.tasks = tasks if tasks is not None else {"summary": "Two follow-ups.", "tasks": [
            {"task": "Call Sam", "owner": None, "due": "tomorrow", "quote": "call Sam tomorrow"},
            {"task": "Send the invoice", "owner": "Priya", "due": "by Friday", "quote": "send the invoice by Friday"}]}
        self.calls = 0

    def transcribe(self, audio, filename, mime, model="m", language="en"):
        self.calls += 1
        return {"text": self.text, "duration": 7.4, "segments": self.segs if self.segs is not None else [{"no_speech_prob": 0.01}]}

    def chat_json(self, messages, models=None, max_tokens=1200):
        return self.tasks, "openai/gpt-oss-20b", {"total_tokens": 321}


class WerTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Hello, World! It's 3-way.", spell_numbers=False), "hello world it's way")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("a b c".split(), "a x c d".split()), 2)
        self.assertEqual(utterance_errors("The cat sat", "the cat sat."), (0, 3))
        self.assertEqual(utterance_errors("one two three four", "one three four"), (1, 4))

    def test_bootstrap_brackets_point_estimate(self):
        pairs = [(1, 10), (0, 12), (2, 8), (0, 9), (1, 11)] * 4
        lo, hi = bootstrap_ci(pairs)
        self.assertLess(lo, 0.1)
        self.assertGreater(hi, 0.05)
        self.assertLessEqual(lo, hi)


class CleanTests(unittest.TestCase):
    def test_grounding_and_drops(self):
        out = clean(FakeClient().tasks, "Remind me to call Sam tomorrow and send the invoice by Friday")
        t = out["tasks"]
        self.assertTrue(all(x["grounded"] for x in t))
        self.assertIsNone(t[1]["owner"])          # "Priya" was never said
        self.assertEqual(t[1]["removed"], ["owner"])
        self.assertEqual(t[0]["due"], "tomorrow")

    def test_invented_quote_is_flagged(self):
        out = clean({"summary": "x", "tasks": [{"task": "Book flights", "quote": "book the flights to Paris"}]}, "buy milk")
        self.assertFalse(out["tasks"][0]["grounded"])

    def test_garbage_does_not_raise(self):
        for bad in (None, [], {"tasks": "no"}, {"tasks": [1, {"task": ""}, {"task": "ok"}]}):
            out = clean(bad, "text")
            self.assertIn("tasks", out)
        self.assertEqual(len(clean({"tasks": [{"task": "t"}] * 50}, "t")["tasks"]), 12)


class AdvancedTests(unittest.TestCase):
    def test_priority_only_from_the_speakers_words(self):
        parsed = {"summary": "", "tasks": [
            {"task": "Send the invoice", "quote": "send the invoice asap"},
            {"task": "Book flights", "quote": "book the flights"}]}
        t = clean(parsed, "please send the invoice asap and book the flights")["tasks"]
        self.assertEqual([x["priority"] for x in t], ["high", None])

    def test_model_claimed_priority_is_ignored(self):
        parsed = {"tasks": [{"task": "Call Sam", "quote": "call Sam", "priority": "high"}]}
        self.assertIsNone(clean(parsed, "call Sam tomorrow")["tasks"][0]["priority"])

    def test_ungrounded_urgent_quote_gets_no_priority(self):
        parsed = {"tasks": [{"task": "Pay rent", "quote": "pay the rent urgently"}]}
        self.assertIsNone(clean(parsed, "buy milk")["tasks"][0]["priority"])

    def test_segment_confidence(self):
        segs = segment_info({"segments": [
            {"start": 0, "end": 2, "text": " hello there", "avg_logprob": -0.2, "compression_ratio": 1.1},
            {"start": 2, "end": 4, "text": " mumbled bit", "avg_logprob": -1.2, "compression_ratio": 1.1},
            {"start": 4, "end": 5, "text": "  ", "avg_logprob": -0.1}]})
        self.assertEqual([(s["text"], s["low"]) for s in segs], [("hello there", False), ("mumbled bit", True)])


class SniffTests(unittest.TestCase):
    def test_types(self):
        self.assertEqual(sniff_audio(WEBM)[1], "audio/webm")
        self.assertEqual(sniff_audio(b"OggS" + b"0" * 20)[1], "audio/ogg")
        self.assertEqual(sniff_audio(b"RIFF\0\0\0\0WAVEfmt ")[1], "audio/wav")
        self.assertEqual(sniff_audio(b"\0\0\0\x20ftypM4A ")[1], "audio/mp4")
        self.assertIsNone(sniff_audio(b"MZ\x90\x00 not audio"))

    def test_silence(self):
        self.assertFalse(speech_found({"segments": [{"no_speech_prob": 0.9}]}))
        self.assertTrue(speech_found({"segments": [{"no_speech_prob": 0.9}, {"no_speech_prob": 0.1}]}))


class AppTests(unittest.TestCase):
    def test_happy_path(self):
        code, out = App(FakeClient()).transcribe(WEBM, "1.1.1.1")
        self.assertEqual(code, 200)
        self.assertEqual(len(out["tasks"]), 2)
        self.assertEqual(out["audio_seconds"], 7.4)
        self.assertIn("total", out["seconds"])

    def test_unconfigured(self):
        self.assertEqual(App(None).transcribe(WEBM, "ip")[0], 503)
        self.assertEqual(App(None).tasks_from_text("a b c d e", "ip")[0], 503)

    def test_validation_does_not_spend_quota(self):
        c = FakeClient()
        app = App(c, Limits(per_ip_per_hour=1))
        self.assertEqual(app.transcribe(b"", "ip")[0], 400)
        self.assertEqual(app.transcribe(b"not audio at all", "ip")[0], 400)
        self.assertEqual(app.transcribe(WEBM + b"0" * 1_600_000, "ip")[0], 413)
        self.assertEqual(c.calls, 0)
        self.assertEqual(app.transcribe(WEBM, "ip")[0], 200)     # the one allowed call is still available

    def test_rate_limits(self):
        app = App(FakeClient(), Limits(per_ip_per_hour=2, daily_cap=3))
        self.assertEqual([app.transcribe(WEBM, "a")[0] for _ in range(3)], [200, 200, 429])
        self.assertEqual(app.transcribe(WEBM, "b")[0], 200)
        self.assertEqual(app.transcribe(WEBM, "c")[0], 429)       # daily cap of 3 reached

    def test_silence_is_reported_not_invented(self):
        app = App(FakeClient(text="Thank you.", segs=[{"no_speech_prob": 0.95}]))
        code, out = app.transcribe(WEBM, "ip")
        self.assertEqual((code, out["empty"], out["tasks"]), (200, True, []))

    def test_text_route(self):
        code, out = App(FakeClient()).tasks_from_text("call Sam tomorrow and send the invoice", "ip")
        self.assertEqual(code, 200)
        self.assertEqual(App(FakeClient()).tasks_from_text("hi", "ip")[0], 400)
        self.assertEqual(App(FakeClient()).tasks_from_text("x " * 3000, "ip")[0], 400)

    def test_provider_errors_are_user_safe(self):
        class Boom(FakeClient):
            def transcribe(self, *a, **k):
                raise GroqError("secret internals", "quota")
        code, out = App(Boom()).transcribe(WEBM, "ip")
        self.assertEqual(code, 503)
        self.assertNotIn("secret", json.dumps(out))


class HttpTests(unittest.TestCase):
    def test_roundtrip(self):
        srv = Server(("127.0.0.1", 0), make_handler(App(FakeClient())))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            c = HTTPConnection("127.0.0.1", srv.server_address[1])
            c.request("GET", "/health"); self.assertEqual(c.getresponse().read(), b'{"ok": true}')
            c.request("POST", "/transcribe", WEBM, {"Content-Type": "audio/webm"})
            r = c.getresponse(); self.assertEqual(r.status, 200); self.assertEqual(len(json.loads(r.read())["tasks"]), 2)
            c.request("POST", "/tasks", json.dumps({"text": "call Sam tomorrow please"}), {"Content-Type": "application/json"})
            self.assertEqual(c.getresponse().status, 200)
            c.request("OPTIONS", "/transcribe"); self.assertEqual(c.getresponse().status, 204)
            c.request("POST", "/nope", b"{}"); self.assertEqual(c.getresponse().status, 404)
        finally:
            srv.shutdown()


class GroqTests(unittest.TestCase):
    def test_retries_then_quota(self):
        seq = [FakeResp(429, headers={"retry-after": "1"}), FakeResp(429), FakeResp(429)]
        slept = []
        g = Groq("k", post=lambda *a, **k: seq.pop(0), sleep=slept.append)
        with self.assertRaises(GroqError) as cm:
            g.transcribe(b"x", "a.webm", "audio/webm")
        self.assertEqual(cm.exception.kind, "quota")
        self.assertEqual(len(slept), 2)

    def test_recovers_after_429(self):
        seq = [FakeResp(429), FakeResp(200, {"text": "hi"})]
        g = Groq("k", post=lambda *a, **k: seq.pop(0), sleep=lambda s: None)
        self.assertEqual(g.transcribe(b"x", "a.webm", "audio/webm")["text"], "hi")

    def test_chat_json_falls_back_and_parses(self):
        calls = []
        def post(url, headers, timeout, json=None, **k):
            calls.append(json["model"])
            if len(calls) == 1:
                return FakeResp(200, {"choices": [{"message": {"content": "not json"}}]})
            return FakeResp(200, {"choices": [{"message": {"content": '{"tasks": []}'}}], "usage": {"total_tokens": 9}})
        parsed, model, usage = Groq("k", post=post, sleep=lambda s: None).chat_json([{"role": "user", "content": "x"}])
        self.assertEqual((parsed, calls[0] != model, usage["total_tokens"]), ({"tasks": []}, True, 9))

    def test_no_key(self):
        with self.assertRaises(GroqError):
            Groq("")


class BenchTests(unittest.TestCase):
    def test_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            sample = [("u1", b"a", "the cat sat", 3.0), ("u2", b"b", "one two three four", 4.0)]
            class C:
                n = 0
                def transcribe(self, raw, fn, mime, model):
                    C.n += 1
                    return {"text": {b"a": "the cat sat.", b"b": "one three four"}[raw]}
            self.assertEqual(bench.run_config("turbo-clean", 2, 1, C(), out, pause=0, sample=sample), 2)
            self.assertEqual(bench.run_config("turbo-clean", 2, 1, C(), out, pause=0, sample=sample), 0)   # resumed: nothing re-sent
            self.assertEqual(C.n, 2)
            s = bench.summarize(out)["turbo-clean"]
            self.assertEqual((s["utterances"], s["words"], s["errors"], s["perfect"]), (2, 7, 1, 1))
            self.assertAlmostEqual(s["wer"], 14.29, places=2)

    def test_config_names(self):
        self.assertEqual(bench.parse_config("v3-other"), ("whisper-large-v3", "other"))
        with self.assertRaises(ValueError):
            bench.parse_config("x-clean")


if __name__ == "__main__":
    unittest.main()
