# Day 3 · Memory + the capstone (~75–90 min)

> Day 2 gave you a graph you control. Day 3 gives it a memory and a conscience:
> it remembers conversations, remembers *users* across conversations, and pauses
> for a human before doing anything risky. These are the last steps from "demo"
> to "product".

## Files

| File | Idea | Time |
|------|------|------|
| `04_short_term_memory.py` | **Checkpointer + thread_id** — per-conversation memory | 15 min |
| `05_long_term_memory.py` | **Store** — facts that persist *across* conversations | 12 min |
| `06_human_in_the_loop.py` | **interrupt / resume** — pause, approve, continue | 8 min (optional) |

```bash
python 03_memory/04_short_term_memory.py
python 03_memory/05_long_term_memory.py
python 03_memory/06_human_in_the_loop.py
```

Each file has a concept demo plus 2–3 "playing around" variations, with comments
at the top of every file and function. Read, then tweak.

## The one-screen summary

```
Checkpointer  → remembers ONE conversation        key: thread_id     (short-term)
Store         → remembers facts ACROSS conversations key: user_id      (long-term)
interrupt()   → pause the graph and wait for a human; resume with Command(resume=...)
```

- **Short-term** = "what did we say earlier in *this* chat?" → `compile(checkpointer=...)`,
  invoke with `{"configurable": {"thread_id": ...}}`.
- **Long-term** = "what do I know about *this user*, ever?" → `compile(store=...)`,
  a node reads/writes a namespaced key-value store with the `user_id`.
- **Human-in-the-loop** = a node calls `interrupt(...)`; the run pauses; you
  inspect it; `Command(resume=...)` continues from the exact same spot.
  (Needs a checkpointer + thread_id.)

## Tie it back to the capstone

The Day 2 [Support Assistant](../02_langgraph/capstone/) already accepts a
checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver
app = build_support_graph(checkpointer=InMemorySaver())
app.invoke({"messages": [...]}, {"configurable": {"thread_id": "ticket-42"}})
```

Now a follow-up ("any update on that refund?") on the same `thread_id` remembers
the ticket. Add a `store` for per-customer history, and wrap the (suggested)
`create_refund` tool in an `interrupt` approval gate — and your capstone is a
product, not a demo.
