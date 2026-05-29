# Adaptive & Agentic RAG

The pipelines in this chapter look at the question and *decide* what to do. They are the difference between a static RAG demo and a production agent.

## The four decisions

Every agentic RAG system answers four implicit questions:

1. **Should I retrieve at all?** (Adaptive RAG, Self-RAG)
2. **Is what I retrieved any good?** (CRAG, Self-RAG, Lynx)
3. **Do I need to retrieve again?** (LangGraph agents)
4. **Which tool should I call?** (MCP, LlamaIndex Workflows)

A useful agent has explicit machinery for each.

## The recipes

- **Self-Reflective Retrieval (24)** — model emits reflection tokens; decides per-chunk usefulness.
- **CRAG (25)** — lightweight evaluator scores retrieved passages; bad scores trigger a web fallback.
- **Adaptive Routing (26)** — classifier picks {no retrieval, single retrieve, multi-hop}.
- **Speculative RAG (27)** — small drafter generates parallel candidates from different retrieval subsets; large verifier picks. Cuts latency by ~50 %.
- **LangGraph Agentic RAG (28)** — stateful cyclic graph; conditional edges retry retrieval until the critic is satisfied.
- **MCP Tool Retrieval (29)** — agent dispatches to remote retrieval tools via the Model Context Protocol.
- **DSPy Compiled RAG (30)** — write a signature, let the optimizer find the best prompts and few-shot examples.

## Production rules

- **Cap loops.** Recipe 28 hard-caps at 3 retries. Without it, a confused critic loops forever.
- **Trace everything.** Phoenix or LangSmith. Agents fail in non-obvious ways; traces are how you find out why.
- **Measure latency, not just quality.** Recipe 27 exists because users feel 500 ms; they tolerate 200 ms.
- **Keep tools small.** Many agents that look like they need 20 tools really need 3 that compose.

## When agents are wrong

If a flat retrieval + rerank + answer pipeline hits your metric targets, stay there. Agentic loops add latency, cost, and failure modes (infinite loops, off-distribution tool calls, prompt-injection surface area) for a quality lift that is often smaller than what a better embedding model would give you.
