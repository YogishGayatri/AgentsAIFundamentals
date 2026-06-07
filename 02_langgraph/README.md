# Day 2 · LangGraph — rebuild the loop as a graph you control

> Yesterday `create_agent` ran the loop for you. Today you build the loop —
> because once you can see it as a **graph**, you can change it: branch it, insert
> steps, inspect it, resume it. By the end you own the three mechanics that every
> agent architecture is made of.

## The three mechanics (and the patterns they unlock)

| File | Mechanic | Pattern it is |
|------|----------|---------------|
| `01_state_nodes_reducers.py` | State, nodes, **sequential edges**, reducers | the sequential pipeline |
| `02_conditional_edges_router.py` | **Conditional edges** | the router |
| `03_agent_loop_as_graph.py` | The model⇄tools **cycle** | ReAct / tool-loop (= `create_agent`) |

Run them in order:

```bash
python 02_langgraph/01_state_nodes_reducers.py
python 02_langgraph/02_conditional_edges_router.py
python 02_langgraph/03_agent_loop_as_graph.py
```

Each file has a concept demo plus 2–3 "playing around" variations. The comments
at the top of every file and function explain the idea — read, then tweak.

## Mental model

```
State   = a shared dict every node reads/writes
Node    = state in  -> partial state out
Edge    = after node A, go to B            (sequential)
Cond.   = after A, a function PICKS where to go next   (router / cycle)
Reducer = HOW an update merges (overwrite by default; append with a reducer)
Compile = freeze it into a runnable: .invoke() / .stream()
```

Every LangGraph agent — including `create_agent` — is just these pieces arranged.
File 03 proves it: we hand-build `create_agent` and then swap in the prebuilt
`ToolNode` / `tools_condition` to show they're the same thing.

## Capstone

After the three files, build the **[Support Triage & Resolution Assistant](capstone/)**
— one graph that uses all three mechanics together (triage → route → specialist⇄tools
→ respond). See [`capstone/README.md`](capstone/README.md).

## What's next

Day 3 adds the two things that turn this into a product: **memory**
(checkpointer + store) and a **human approval** step (interrupt / resume).
See [`../03_memory/`](../03_memory/).
