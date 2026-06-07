# Capstone · Support Triage & Resolution Assistant

The Day 2 capstone. One graph that uses **all three mechanics** you learned today —
sequential edges, a conditional router, and the model⇄tools cycle — to do
something real: take a raw customer message and produce a resolved, customer-ready
reply, escalating when it matters.

> This is the moment it clicks: `create_agent` was *one* node-pattern (the cycle).
> A real assistant is a **graph of patterns** — triage, route, act, respond — and
> now you can build that yourself.

---

## What it does

```
Customer message
   ↓ triage        understand it  → category, urgency, needs_lookup, summary
   ↓ route         need data?      → yes: go work the tools / no: answer directly
   ↓ specialist ⇄ tools           the agent cycle: look up orders/billing/help
   ↓ respond       write the reply → friendly answer (+ escalation banner if urgent)
Resolution
```

It handles four kinds of tickets out of the box (`billing`, `technical`,
`account`, `general`) and decides per-ticket whether it even needs to touch a tool.

## The graph

```
        START
          │
          ▼
       triage ───────────► route_after_triage  ── needs_lookup? ──┐
                                   │ yes                           │ no
                                   ▼                               ▼
                              specialist ◄────┐                 respond ──► END
                                   │          │                    ▲
                            should_continue   │ loop back          │
                              │ tools   │ else │                    │
                              ▼         └──────┴── tools ───────────┘
                            tools  ── results ──► specialist
```

Print it yourself:
```bash
python 02_langgraph/capstone/main.py --draw     # Mermaid source
```

### Where each Day-2 mechanic lives

| Mechanic (today's file) | In the capstone |
|---|---|
| Sequential edges (File 01) | `START → triage`, `respond → END` |
| Conditional router (File 02) | `route_after_triage` — work the tools, or answer directly |
| The agent cycle (File 03) | `specialist ⇄ tools` until no more tool calls |
| Structured output (Day 1) | `triage` fills the typed `Triage` schema that the router switches on |

## File tour

| File | Role |
|------|------|
| `schemas.py` | The `Triage` Pydantic contract the router switches on |
| `state.py`   | `SupportState` — the shared dict flowing through the graph |
| `tools.py`   | Canned support tools (order status, billing, help center) |
| `nodes.py`   | Every node + both router functions — **the brain; read this first** |
| `graph.py`   | Wires nodes into the graph (read the edges as a sentence) |
| `main.py`    | Runner: sample tickets, `--chat`, `--draw` |

## Run it

From the **repo root**, with your `.env` set (Groq by default):

```bash
python 02_langgraph/capstone/main.py            # run the 4 sample tickets
python 02_langgraph/capstone/main.py --chat     # type your own
python 02_langgraph/capstone/main.py --draw     # print the graph
```

You'll see a live trace per ticket, e.g.:

```
TICKET: I was charged twice for ACC-9 this month — refund the duplicate ASAP!
  [triage]     category=billing urgency=5 needs_lookup=True
  [specialist] requested ['get_billing_summary']
  [tools]      -> ACC-9 last charges: $20 on Jun 1, $20 on Jun 1 (DUPLICATE)...
  [specialist] no tools, finishing
RESOLUTION:
  ⚠️ [URGENT — flagged for a human agent]
  You're right — I can see a duplicate $20 charge on Jun 1 for ACC-9 ...
```

## Extend it (suggested exercises)

1. **Add a branch.** Give `general` tickets their own node that never uses tools,
   and route to it explicitly instead of folding it into `respond`.
2. **Add a tool.** Add `create_refund(account_id, amount)` and let the billing
   path call it — then guard it behind the Day-3 human-in-the-loop approval.
3. **Per-category tools.** Bind only the relevant tools per category (billing
   tools for billing tickets) by building the model inside the node.
4. **Add memory (Day 3).** Pass a `checkpointer` to `build_support_graph(...)` and
   a `thread_id` so a follow-up message ("any update on that refund?") remembers
   the ticket.
5. **A verdict gate.** Before `respond`, add a node that double-checks the
   resolution actually addresses the `summary`, looping back if not.

## How this maps back to the week

- Day 1: model, messages, tools, structured output, `create_agent`.
- Day 2: you opened the loop into a graph — and here, composed several patterns
  into one assistant.
- Day 3: you'll give this graph memory and a human approval step — the last two
  things that separate a demo from a product.
