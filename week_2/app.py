import io
import wave

import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Pretrained Hugging Face pipelines, loaded lazily on first use and cached.
_pipelines = {}

PIPELINE_SPECS = {
    "sentiment": ("sentiment-analysis",
                  "distilbert-base-uncased-finetuned-sst-2-english"),
    "summary": ("summarization", "sshleifer/distilbart-cnn-12-6"),
    "speech": ("automatic-speech-recognition", "openai/whisper-tiny.en"),
    "caption": ("image-to-text", "Salesforce/blip-image-captioning-base"),
}


def get_pipeline(name):
    if name not in _pipelines:
        from transformers import pipeline
        task, model = PIPELINE_SPECS[name]
        _pipelines[name] = pipeline(task, model=model)
    return _pipelines[name]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify(error="no text provided"), 400
    try:
        r = get_pipeline("sentiment")(text)[0]
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(label=r["label"], score=round(r["score"], 4))


@app.route("/api/summary", methods=["POST"])
def api_summary():
    text = request.json.get("text", "").strip()
    n_words = len(text.split())
    if n_words < 20:
        return jsonify(error="text too short to summarize "
                             "(give it at least a couple of sentences)"), 400
    try:
        r = get_pipeline("summary")(text, max_length=142,
                                    min_length=12, truncation=True)[0]
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(summary=r["summary_text"].strip(), input_words=n_words)


@app.route("/api/speech", methods=["POST"])
def api_speech():
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="no audio file uploaded"), 400
    try:
        with wave.open(io.BytesIO(f.read()), "rb") as w:
            rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        r = get_pipeline("speech")({"raw": audio, "sampling_rate": rate})
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(transcript=r["text"].strip(),
                   seconds=round(len(audio) / rate, 1))


@app.route("/api/caption", methods=["POST"])
def api_caption():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="no image file uploaded"), 400
    try:
        from PIL import Image
        img = Image.open(f.stream).convert("RGB")
        r = get_pipeline("caption")(img, max_new_tokens=30)
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(caption=r[0]["generated_text"].strip(),
                   size=f"{img.width}x{img.height}")


@app.route("/health")
def health():
    return jsonify(status="ok", loaded=sorted(_pipelines))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
