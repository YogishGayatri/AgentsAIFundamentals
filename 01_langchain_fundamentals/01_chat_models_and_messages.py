"""
1. Chat models + the four messages as objects  (~12 min)

Punchline first: the four roles and the chat template you saw in foundations are
now just Python objects. A conversation is a LIST of message objects; you call
the model on the list; you get an AIMessage back.

    SystemMessage / HumanMessage / AIMessage / ToolMessage
        = system / user / assistant / tool  (the wire-format roles), now typed.

Run:  python 01_langchain_fundamentals/01_chat_models_and_messages.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from providers import get_chat_model

# temperature=0 -> as deterministic as the model allows. You want this for
# tool-using agents and for demos that should behave the same every run.
model = get_chat_model(temperature=0)

# A conversation is just a list of message objects.
messages = [
    SystemMessage("You are a helpful assistant. Be concise."),
    HumanMessage("In one sentence, what is the capital of France known for?"),
]

reply = model.invoke(messages)   # one trip through the LLM: messages in, one AIMessage out

print("Reply text:", reply.content)
print("Reply type:", type(reply).__name__)   # AIMessage

# Teaching note:
# LangChain applied the chat template and special tokens for you on the way to
# the provider -- exactly the layer we showed was happening on the server in
# foundations. You think in objects; the wire format is handled underneath.
#
# Because the reply is itself a message, you keep a conversation going simply by
# appending it (and the next human turn) back onto the list:
messages.append(reply)
messages.append(HumanMessage("And name one famous museum there."))
followup = model.invoke(messages)
print("Follow-up:", followup.content)
