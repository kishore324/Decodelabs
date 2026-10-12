# Project 1: Rule-Based AI Chatbot

DecodeLabs | Artificial Intelligence | Industrial Training Kit | Batch 2026

A deterministic chatbot built on control flow and logic, following the IPO model (Input -> Process -> Output).

## Run

```bash
python chatbot.py
```

## Spec checklist

| Requirement | Implementation |
|---|---|
| Input loop | `while True:` in `main()` |
| Sanitization | `sanitize()` lowercases, strips and collapses whitespace |
| Knowledge base | `INTENTS` dict with 7 intents, flattened into `RESPONSES` |
| Fallback | `RESPONSES.get(user_input, FALLBACK)` |
| Exit strategy | `exit`, `quit`, `bye`, `goodbye` trigger `break` |

## Example

```
You:   HELLO
DecodeBot: Hi there! I'm DecodeBot. How can I help you today?
You: what can you do
DecodeBot: I can chat about greetings, ...
You: blah
DecodeBot: I do not understand. Try 'help' to see what I can do.
You: exit
DecodeBot: Goodbye! Have a great day.
```

## Ideas to extend (from the Conclusion slide)

- Expand the vocabulary with more intents
- Add nested conditions for smarter responses
- Give the bot a unique personality
