import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from providers import get_chat_model

model = get_chat_model(temperature=0)

# Messages
messages = [
    SystemMessage("You are a helpful assistant. Be concise."),
    HumanMessage("In one sentence, what is the capital of France known for?"),
]

reply = model.invoke(messages)   

print("Reply text:", reply.content)
#print("Reply", reply)

print("Reply type:", type(reply).__name__)

messages.append(reply)
messages.append(HumanMessage("And name one famous museum there."))
followup = model.invoke(messages)
print("Follow-up:", followup.content)
messages.append(HumanMessage("Tell me more about tourist sites in France."))

"""
print("Streaming: ", end="", flush=True)
for chunk in model.stream("Tell me more about tourist sites in France."):
    print(chunk.content, end="", flush=True)
print()

"""

#print(model.invoke([HumanMessage("And name one famous museum there.")]).content)
#print(model.invoke([("system", "Be concise."), ("human", "Capital of France?")]).content)
#print(model.invoke([{"role":"system", "content":"Be concise and short."},{"role":"human", "content":"Capital of France"}]).content)
