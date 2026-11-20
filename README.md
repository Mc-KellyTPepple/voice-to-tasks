# Voice to Tasks

Speak for up to 30 seconds. You get the transcript, a one-line summary and a to-do list, with each task tied to the exact words you said.

```
browser microphone → POST /transcribe → Whisper (speech to text) → text model (summary + tasks) → checks against the transcript
```

## What it checks

- **Grounded tasks.** The model must quote the words behind every task. A task whose quote is not in the transcript is marked *not grounded*, and an owner or due date that was never said is removed.
- **Silence.** Whisper invents text for silent clips. Clips where every segment is probably not speech are reported as empty instead of turned into tasks.
- **Limits.** Per-visitor and daily request limits, a 1.5 MB audio limit, file-type check before any model call, audio held in memory only, nothing logged.

## Measured accuracy

Word error rate on random samples of LibriSpeech test-clean (read speech) and test-other (harder speakers and recordings), run by GitHub Actions. See `evals/results/` for every transcript and `summary.json` for the numbers. The table is filled by the run, not typed in.

| Configuration | Clips | WER | 95% interval |
|---|---|---|---|
| _pending first run_ | | | |

Method: fixed sample (seed 1) of utterances up to 30 s, original FLAC sent to the API, text lowercased with digits spelled out and punctuation removed, corpus-level WER (total errors / total reference words), bootstrap interval over utterances. LibriSpeech is audiobook speech, so it is easier than a rushed voice note on a phone; treat the number as a floor for this task, not a promise.

## Run it

```
pip install -e .
GROQ_API_KEY=... python -m voicetotasks serve      # http://localhost:7860
python -m unittest discover -s tests -t .
pip install -e ".[eval]" && GROQ_API_KEY=... python -m voicetotasks bench --n 50
```

`POST /transcribe` takes raw audio (webm, ogg, mp4, wav, mp3, flac). `POST /tasks` takes `{"text": "..."}` for visitors without a microphone.

Deployment: `render.yaml`. Set `GROQ_API_KEY` as a secret; the service starts without it and answers 503 until it is added.
