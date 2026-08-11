"""
 DEMO 2 : Regex (Pattern) Rule-Based Chatbot
---------------------------------------------------
Better than keywords: patterns can CAPTURE information
   "my name is Ravi"   -> captures "Ravi"
   "track order 4521"  -> captures "4521"

Run:  python 2_regex_bot.py
"""

import re
import random

# pattern  ->  list of possible replies ( {} = captured value )
PATTERNS = [
    (r"\b(hi|hello|hey)\b",
     ["Hi there! I am ShopBot.", "Hello! How can I help you today?"]),

    (r"my name is ([a-z]+)",
     ["Nice to meet you, {0}!", "Welcome, {0}!"]),

    (r"track (?:my )?order (\d+)",
     ["Order #{0} is packed and will arrive in 2 days."]),

    (r"\b(refund|return)\b",
     ["Returns are accepted within 7 days of delivery."]),

    (r"(?:what|which) products",
     ["We sell laptops, phones and accessories."]),

    (r"\b(bye|quit|exit)\b",
     ["Bye! Thanks for visiting.", "See you soon!"]),
]

FALLBACK = ["Could you rephrase that?", "I am not sure I understand."]


def get_response(message: str) -> str:
    text = message.lower().strip()
    for pattern, replies in PATTERNS:
        match = re.search(pattern, text)
        if match:
            reply = random.choice(replies)
            # insert captured groups, if the reply needs them
            return reply.format(*[g.title() for g in match.groups()])
    return random.choice(FALLBACK)


def chat():
    print("ShopBot: Hello! (type 'bye' to exit)")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        print("ShopBot:", get_response(user))
        if re.search(r"\b(bye|quit|exit)\b", user.lower()):
            break


if __name__ == "__main__":
    chat()
