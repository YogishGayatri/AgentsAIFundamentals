"""
hello_llm.py — the smallest possible "I called an LLM from Python" program.

Run:  python 00_setup/hello_llm.py

That's the entire goal of Day -1: text goes in, text comes out, from real code.
Notice we don't import ChatGroq directly — we ask providers.get_chat_model() for
a model. That one habit is why you'll be able to switch providers later by
editing .env instead of editing code.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import get_chat_model


def main():
    model = get_chat_model()  # Groq by default; honors PROVIDER in .env

    prompt = "In one friendly sentence, tell me what a large language model is."
    print(f"\nYou:   {prompt}")

    reply = model.invoke(prompt)   # one trip through the LLM
    print(f"Model: {reply.content}\n")


if __name__ == "__main__":
    main()
