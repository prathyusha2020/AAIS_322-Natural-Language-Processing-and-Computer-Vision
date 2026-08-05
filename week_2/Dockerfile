FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY templates ./templates

# Download all four pretrained models at build time so the container
# starts instantly and needs no network at runtime.
RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english'); \
    pipeline('summarization', model='sshleifer/distilbart-cnn-12-6'); \
    pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en'); \
    pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')"

EXPOSE 5000
CMD ["python", "app.py"]
