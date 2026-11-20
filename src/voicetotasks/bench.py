"""Word error rate of the speech models on LibriSpeech test-clean and test-other.

  python -m voicetotasks bench --configs turbo-clean,turbo-other,v3-clean,v3-other --n 50 --seed 1

A fixed random sample (seed 1) of utterances up to 30 s long is sent to the Groq API exactly as the product would
send it (the original FLAC bytes). Results are appended to evals/results/<config>.jsonl so an interrupted run resumes.
`summary` recomputes everything from those files; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from .groq import Groq, GroqError
from .wer import bootstrap_ci, corpus_wer, utterance_errors

RESULTS = Path("evals/results")
MODELS = {"turbo": "whisper-large-v3-turbo", "v3": "whisper-large-v3"}
SUBSETS = {"clean": "clean", "other": "other"}
MAX_SECONDS = 30.0


def parse_config(name: str) -> tuple[str, str]:
    model, _, subset = name.partition("-")
    if model not in MODELS or subset not in SUBSETS:
        raise ValueError(f"unknown config {name!r}; use e.g. turbo-clean, v3-other")
    return MODELS[model], SUBSETS[subset]


def load_sample(subset: str, n: int, seed: int):
    """Yield n utterances (id, flac bytes, reference text, seconds). Needs `pip install .[eval]`."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("openslr/librispeech_asr", subset, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False)).shuffle(seed=seed, buffer_size=3000)
    got = 0
    for row in ds:
        raw = row["audio"]["bytes"]
        info = sf.info(io.BytesIO(raw))
        if info.duration > MAX_SECONDS:
            continue
        yield str(row["id"]), raw, row["text"], round(info.duration, 2)
        got += 1
        if got >= n:
            return


def read_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r.get("hypothesis") is not None}


def run_config(name: str, n: int, seed: int, client: Groq, out_dir: Path = RESULTS, pause: float = 3.2, sample=None) -> int:
    model, subset = parse_config(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    done = read_done(path)
    fresh = 0
    for uid, raw, ref, secs in (sample if sample is not None else load_sample(subset, n, seed)):
        if uid in done:
            continue
        t0 = time.monotonic()
        try:
            out = client.transcribe(raw, f"{uid}.flac", "audio/flac", model)
        except GroqError as exc:
            if exc.kind == "quota":
                print(f"{name}: provider quota reached after {len(done) + fresh} clips; resume later", flush=True)
                break
            print(f"{name}: {uid} failed ({exc})", flush=True)
            continue
        latency = round(time.monotonic() - t0, 2)
        hyp = (out.get("text") or "").strip()
        errs, words = utterance_errors(ref, hyp)
        row = {"id": uid, "seconds": secs, "latency": latency, "reference": ref, "hypothesis": hyp, "errors": errs, "words": words}
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        fresh += 1
        time.sleep(pause)
    return fresh


def summarize(out_dir: Path = RESULTS) -> dict:
    configs = {}
    for path in sorted(out_dir.glob("*.jsonl")):
        rows = list(read_done(path).values())
        if not rows:
            continue
        pairs = [(r["errors"], r["words"]) for r in rows]
        lo, hi = bootstrap_ci(pairs)
        lat = sorted(r["latency"] for r in rows)
        secs = sum(r["seconds"] for r in rows)
        worst = sorted(rows, key=lambda r: r["errors"] / max(r["words"], 1), reverse=True)[:5]
        configs[path.stem] = {
            "utterances": len(rows), "words": sum(r["words"] for r in rows), "errors": sum(r["errors"] for r in rows),
            "wer": round(corpus_wer(pairs) * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
            "perfect": sum(1 for r in rows if r["errors"] == 0),
            "audio_seconds": round(secs, 1), "median_latency": round(statistics.median(lat), 2),
            "p90_latency": round(lat[min(len(lat) - 1, int(len(lat) * 0.9))], 2),
            "speed_x_realtime": round(secs / max(sum(r["latency"] for r in rows), 0.01), 1),
            "worst": [{"id": r["id"], "errors": r["errors"], "words": r["words"], "reference": r["reference"], "hypothesis": r["hypothesis"]} for r in worst],
        }
    return configs


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="voicetotasks bench")
    ap.add_argument("--configs", default="turbo-clean,turbo-other,v3-clean,v3-other")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summary_only:
        client = Groq(os.environ.get("GROQ_API_KEY", ""))
        for name in a.configs.split(","):
            print(f"{name}: {run_config(name.strip(), a.n, a.seed, client)} new clips", flush=True)
    summary = summarize()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        print(f"{k}: WER {v['wer']}% (95% CI {v['ci_lo']}–{v['ci_hi']}) over {v['utterances']} clips, median {v['median_latency']}s", flush=True)
