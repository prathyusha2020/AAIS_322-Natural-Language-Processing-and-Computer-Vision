"""
 DEMO 4 : Dialogflow Fulfillment Webhook (Flask)
--------------------------------------------------------
Dialogflow does the UNDERSTANDING (intent + entities).
This webhook does the WORK (database, price calculation, order id).

Dialogflow ES sends a POST request  ->  we read intent + parameters
                                    ->  we return {"fulfillmentText": "..."}

Local test:
    pip install flask
    python 4_dialogflow_webhook.py
    ngrok http 5000
    Paste the https URL + /webhook into Dialogflow > Fulfillment
"""

from flask import Flask, request, jsonify
import random

app = Flask(__name__)

PRICE = {"small": 199, "medium": 299, "large": 399}


@app.route("/webhook", methods=["POST"])
def webhook():
    body = request.get_json(force=True)

    intent = body["queryResult"]["intent"]["displayName"]
    params = body["queryResult"]["parameters"]

    if intent == "order.pizza":
        reply = handle_order(params)
    elif intent == "order.status":
        reply = f"Order #{params.get('order_id')} is out for delivery."
    else:
        reply = "Sorry, I cannot handle that yet."

    return jsonify({"fulfillmentText": reply})


def handle_order(params):
    size = params.get("size", "medium")
    topping = params.get("topping", "cheese")
    price = PRICE.get(size, 299)
    order_id = random.randint(1000, 9999)
    # ---- real project: save to database here ----
    return (f"Your {size} {topping} pizza is confirmed. "
            f"Order #{order_id}, total Rs.{price}. Delivery in 30 minutes.")


@app.route("/", methods=["GET"])
def health():
    return "Webhook is running", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
