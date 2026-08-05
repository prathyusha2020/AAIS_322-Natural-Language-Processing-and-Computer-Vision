# Four Pretrained Models, One Flask App

A single Flask web app that serves four machine-learning features through a
simple browser UI and a JSON/file-upload API:

1. **Sentiment analysis** — type any text, get POSITIVE / NEGATIVE with a confidence score
2. **Text summarization** — paste any paragraph(s), get an abstractive summary
3. **Speech to text** — record from your microphone (or upload an audio file), get a transcript
4. **Image captioning** — upload any image, get a natural-language caption

All four run locally on CPU using pretrained models from Hugging Face — no
API keys, no cloud calls at inference time.

## Models used

| Feature | Model | Task | Size | Hugging Face link |
|---|---|---|---|---|
| Sentiment analysis | DistilBERT (fine-tuned on SST-2) | `sentiment-analysis` | ~260 MB | [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| Summarization | DistilBART (fine-tuned on CNN/DailyMail) | `summarization` | ~1.1 GB | [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) |
| Speech to text | Whisper tiny (English) | `automatic-speech-recognition` | ~150 MB | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) |
| Image captioning | BLIP base | `image-to-text` | ~990 MB | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |

Each model is loaded lazily the first time its feature is used, then kept in
memory. Downloads are cached in `~/.cache/huggingface`, so they only happen
once per machine.

## Run locally

Requirements: Python 3.10+ and ~4 GB of free disk for the model cache.

```bash
# 1. clone and enter the repo
git clone <your-repo-url>
cd <repo>

# 2. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. start the app
python app.py
```

Then open **http://localhost:5000** in Chrome or Edge.

Notes:

- The **first click** on each feature downloads/loads its model — the first
  summarization or caption can take a minute. After that, responses are fast.
- Microphone recording requires `localhost` or HTTPS (a browser security
  rule). It works out of the box at http://localhost:5000. If another app
  (e.g. Zoom) is holding the microphone, the page will tell you.

## Run with Docker

```bash
docker build -t four-models .
docker run -p 5000:5000 four-models
```

All models are downloaded at build time, so the container starts instantly
and needs no network at runtime (image is ~7 GB).

## API

| Route | Method | Body | Returns |
|---|---|---|---|
| `/api/sentiment` | POST | JSON `{"text": "..."}` | `{label, score}` |
| `/api/summary` | POST | JSON `{"text": "..."}` (20+ words) | `{summary, input_words}` |
| `/api/speech` | POST | multipart file field `audio` (16-bit PCM WAV) | `{transcript, seconds}` |
| `/api/caption` | POST | multipart file field `image` (jpg/png/...) | `{caption, size}` |
| `/health` | GET | — | `{status, loaded}` |

Examples:

```bash
curl -X POST localhost:5000/api/sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"this project works great"}'

curl -X POST localhost:5000/api/summary \
  -H 'Content-Type: application/json' \
  -d '{"text":"<paste a paragraph of at least 20 words here>"}'

curl -X POST localhost:5000/api/speech  -F 'audio=@recording.wav'
curl -X POST localhost:5000/api/caption -F 'image=@photo.jpg'
```

The browser UI records microphone audio, converts it to 16 kHz mono WAV in
JavaScript, and posts it to `/api/speech` — so the server only ever has to
parse plain WAV and never needs ffmpeg.

## Project structure

```
app.py               Flask app: 4 API routes + page route, lazy model loading
templates/
  index.html         single-page UI (recording, uploads, fetch calls)
requirements.txt     flask, torch, numpy, transformers, pillow
Dockerfile           container build with models baked in
```
