"""
DecodeLabs | Artificial Intelligence | Project 1
Rule-Based AI Chatbot  ("The Logic Engine")

Architecture: IPO model
    INPUT   -> sanitization & normalization
    PROCESS -> intent matching via dictionary lookup (O(1))
    OUTPUT  -> response generation + feedback loop

Spec checklist
    [x] INPUT LOOP     : continuous 'while' cycle
    [x] SANITIZATION   : handles case & whitespace
    [x] KNOWLEDGE BASE : dictionary with 5+ intents
    [x] FALLBACK       : default response for unknowns (dict.get)
    [x] EXIT STRATEGY  : clean break command
"""

BOT_NAME = "DecodeBot"

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE
# Each intent has a response and a list of trigger phrases (keys).
# These are flattened into one dictionary so lookup is a single O(1) .get().
# ---------------------------------------------------------------------------
INTENTS = {
    "greeting": {
        "triggers": ["hello", "hi", "hey", "good morning", "good evening"],
        "response": "Hi there! I'm DecodeBot. How can I help you today?",
    },
    "how_are_you": {
        "triggers": ["how are you", "how are you?", "how's it going", "what's up"],
        "response": "I'm just a bundle of if-else logic, but I'm running perfectly!",
    },
    "name": {
        "triggers": ["what is your name", "who are you", "your name"],
        "response": "I'm DecodeBot, a rule-based chatbot built at DecodeLabs.",
    },
    "purpose": {
        "triggers": ["what can you do", "help", "what do you do"],
        "response": (
            "I can chat about greetings, tell you about myself, explain what "
            "rule-based AI is, and more. Type 'exit' to leave."
        ),
    },
    "rule_based_ai": {
        "triggers": ["what is rule based ai", "what is rule-based ai", "rule based ai"],
        "response": (
            "Rule-based AI follows explicit, hard-coded instructions. "
            "Input -> Logic -> Output. No mystery, no hallucinations."
        ),
    },
    "thanks": {
        "triggers": ["thanks", "thank you", "thx"],
        "response": "You're welcome!",
    },
    "creator": {
        "triggers": ["who made you", "who created you", "who built you"],
        "response": "I was built by an AI engineer intern at DecodeLabs as Project 1.",
    },
}

# Flatten: trigger phrase -> response (the hash map / dictionary pivot)
RESPONSES = {
    trigger: data["response"]
    for data in INTENTS.values()
    for trigger in data["triggers"]
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}
FALLBACK = "I do not understand. Try 'help' to see what I can do."


# ---------------------------------------------------------------------------
# PHASE 1: INPUT & SANITIZATION
# ---------------------------------------------------------------------------
def sanitize(raw_input: str) -> str:
    """Normalize case and whitespace (also collapses inner extra spaces)."""
    return " ".join(raw_input.lower().strip().split())


# ---------------------------------------------------------------------------
# PHASE 2: PROCESS
# ---------------------------------------------------------------------------
def get_reply(clean_input: str) -> str:
    """Atomic lookup + fallback in a single operation."""
    return RESPONSES.get(clean_input, FALLBACK)


# ---------------------------------------------------------------------------
# PHASE 3: OUTPUT + THE HEARTBEAT (infinite loop)
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"{BOT_NAME}: Hello! Type 'help' for options or 'exit' to quit.")

    while True:
        try:
            raw_input_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Goodbye!")
            break

        clean_input = sanitize(raw_input_text)

        # Ignore empty input
        if not clean_input:
            continue

        # Kill command
        if clean_input in EXIT_COMMANDS:
            print(f"{BOT_NAME}: Goodbye! Have a great day.")
            break

        print(f"{BOT_NAME}: {get_reply(clean_input)}")


if __name__ == "__main__":
    main()
