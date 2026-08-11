#!/bin/bash
# DAY 3 - Deploy the Dialogflow webhook to Google Cloud Run
PROJECT=my-pizza-bot-123

gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# build + deploy in one command
gcloud run deploy pizza-webhook \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated

# copy the printed https URL, add /webhook,
# and paste it in Dialogflow > Fulfillment > Webhook URL
