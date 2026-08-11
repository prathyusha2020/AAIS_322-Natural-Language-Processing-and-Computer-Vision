"""
- DEMO 5 : Talk to your Dialogflow agent from Python
------------------------------------------------------------
Use this when you want the bot inside YOUR OWN app / website
instead of the Dialogflow test console.

Setup:
    pip install google-cloud-dialogflow
    # download the service-account key from Google Cloud console
    export GOOGLE_APPLICATION_CREDENTIALS="key.json"

Run:
    python 5_dialogflow_client.py
"""

import uuid
from google.cloud import dialogflow

PROJECT_ID = "my-pizza-bot-123"      # <-- your Dialogflow project id
LANGUAGE = "en"


def detect_intent(text: str, session_id: str) -> str:
    client = dialogflow.SessionsClient()
    session = client.session_path(PROJECT_ID, session_id)

    text_input = dialogflow.TextInput(text=text, language_code=LANGUAGE)
    query_input = dialogflow.QueryInput(text=text_input)

    response = client.detect_intent(session=session, query_input=query_input)
    result = response.query_result

    print(f"   [intent : {result.intent.display_name} "
          f"| confidence : {result.intent_detection_confidence:.2f}]")
    return result.fulfillment_text


if __name__ == "__main__":
    session_id = str(uuid.uuid4())   # one session = one conversation
    print("Connected to Dialogflow. (type 'bye' to exit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("bye", "exit", "quit"):
            break
        print("Bot:", detect_intent(user, session_id))
