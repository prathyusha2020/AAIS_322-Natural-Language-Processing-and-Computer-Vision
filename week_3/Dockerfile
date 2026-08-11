# DAY 3 - Deploying the webhook to Google Cloud Run
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY 4_dialogflow_webhook.py .
ENV PORT=8080
CMD exec gunicorn --bind :$PORT 4_dialogflow_webhook:app
