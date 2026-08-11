"""
 DEMO 1 : Keyword Rule-Based Chatbot
-------------------------------------------
Idea: keep a dictionary of  keywords -> reply.
If any keyword appears in the user's sentence, send that reply.

Run:  python 1_keyword_bot.py
"""

# ---------------- 1. The rule book ----------------
RULES = {
    ("hi", "hello", "hey"):
        "Hello! I am CampusBot. Ask me about fees, timings or courses.",

    ("fee", "fees", "payment"):
        "Semester fee is Rs. 45,000. Pay online at portal.college.edu/fees",

    ("timing", "timings", "hours", "open"):
        "Office hours are Monday to Friday, 9:00 AM to 5:00 PM.",

    ("course", "courses", "subject"):
        "We offer B.Tech in CSE, AI & DS, ECE and Mechanical.",

    ("bye", "exit", "quit", "goodbye"):
        "Goodbye! Happy learning.",
}

FALLBACK = "Sorry, I did not understand. Try: fees / timings / courses."


# ---------------- 2. The brain ----------------
def get_response(message: str) -> str:
    text = message.lower()
    for keywords, reply in RULES.items():
        if any(word in text for word in keywords):
            return reply
    return FALLBACK

# ---------------- 3. The chat loop ----------------
def chat():
    print("CampusBot: Hello! (type 'bye' to exit)")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        print("CampusBot:", get_response(user))
        if any(w in user.lower() for w in ("bye", "exit", "quit")):
            break


if __name__ == "__main__":
    chat()
