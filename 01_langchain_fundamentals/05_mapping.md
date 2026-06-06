# The mapping, made explicit (~4 min)

This is the one slide that makes the whole day click. Put your raw loop from
foundations on the left, and what you wrote today on the right.

| Raw loop (foundations) | LangChain today |
|---|---|
| "Here's the tool menu" | `bind_tools([...])` / `tools=` |
| Model emits a tool-call intention | `AIMessage.tool_calls` |
| You run the tool | `tool.invoke(args)` |
| Feed the result back | `ToolMessage(result, tool_call_id=...)` |
| The `while` loop | `create_agent` |
| The growing messages list (memory) | the `messages` list — and tomorrow, **LangGraph state** |

The takeaway in one breath:

> You didn't learn a new way to build agents today. You learned the **typed,
> reusable version of the loop you already wrote** — and the easy button that
> runs it for you. Tomorrow we stop hiding the loop and start controlling it.
