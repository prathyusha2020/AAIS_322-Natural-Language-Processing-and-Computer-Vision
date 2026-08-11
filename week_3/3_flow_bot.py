"""
 DEMO 3 : Flow-Based (State Machine) Chatbot
----------------------------------------------------
A keyword bot answers one question at a time.
A FLOW bot remembers WHERE the user is in the conversation
and collects data step by step.

FLOW:
  START -> SIZE -> TOPPING -> ADDRESS -> CONFIRM -> END
Each state:  validates input -> stores it -> moves to next state
Invalid input -> stay in the same state and ask again.

Run:  python 3_flow_bot.py
"""

SIZES = ["small", "medium", "large"]
TOPPINGS = ["cheese", "paneer", "veggie", "chicken"]
PRICE = {"small": 199, "medium": 299, "large": 399}


class PizzaBot:
    def __init__(self):
        self.state = "START"
        self.order = {}          # <-- the "memory" / slots

    # ---------- one message in, one message out ----------
    def reply(self, msg: str) -> str:
        text = msg.lower().strip()

        if text in ("cancel", "restart"):
            self.__init__()
            return "Order cancelled. Say 'hi' to start again."

        handler = getattr(self, "state_" + self.state.lower())
        return handler(text)

    # ---------- one method per state ----------
    def state_start(self, text):
        self.state = "SIZE"
        return "Welcome to PizzaBot! What size would you like? (small/medium/large)"

    def state_size(self, text):
        if text not in SIZES:
            return "Please choose small, medium or large."
        self.order["size"] = text
        self.state = "TOPPING"
        return f"Got it, {text}. Which topping? ({'/'.join(TOPPINGS)})"

    def state_topping(self, text):
        if text not in TOPPINGS:
            return f"Sorry, we have: {', '.join(TOPPINGS)}."
        self.order["topping"] = text
        self.state = "ADDRESS"
        return "Great. What is your delivery address?"

    def state_address(self, text):
        if len(text) < 5:
            return "That address looks too short. Please type the full address."
        self.order["address"] = text.title()
        self.state = "CONFIRM"
        o = self.order
        return (f"Confirm: {o['size']} {o['topping']} pizza to {o['address']} "
                f"for Rs.{PRICE[o['size']]}. (yes/no)")

    def state_confirm(self, text):
        if text in ("yes", "y", "confirm"):
            self.state = "END"
            return "Order placed! Your pizza arrives in 30 minutes."
        if text in ("no", "n"):
            self.__init__()
            return "Order cancelled. Say 'hi' to start again."
        return "Please answer yes or no."

    def state_end(self, text):
        return "Your order is already placed. Type 'restart' for a new order."


def chat():
    bot = PizzaBot()
    print("PizzaBot: Say 'hi' to begin. (type 'cancel' anytime)")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("bye", "exit", "quit"):
            print("PizzaBot: Bye!")
            break
        print("PizzaBot:", bot.reply(user))


if __name__ == "__main__":
    chat()
