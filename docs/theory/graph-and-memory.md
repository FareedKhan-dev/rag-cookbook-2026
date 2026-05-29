# Graph & Memory

Some questions cannot be answered from any single passage. Multi-hop reasoning ("which Cooper-pair-related phenomenon does the same paper that introduced SQUIDs cite as evidence?") needs the structure that connects passages, not just the passages themselves. That structure lives in a graph, and the cookbook treats graph and memory as the same chapter because they share machinery.

## The four recipes

- **Microsoft GraphRAG (31)** — extract triples, build KG, run Leiden community detection, route queries to local (entity) or global (community summary) retrieval. Best for corpus-spanning questions.
- **LightRAG (32)** — dual-level retrieval on the same KG. About 10x cheaper than Microsoft GraphRAG; incremental updates.
- **HippoRAG (33)** — personalized PageRank from query entities. Surfaces intermediate hops that no single chunk contains.
- **Mem0 long-term memory (34)** — extract durable facts from conversation turns, recall by relevance. The production pattern for assistants that talk to users across sessions.

## Where the lift comes from

- **GraphRAG and LightRAG** win on synthesis: "across the corpus, what are the major themes?".
- **HippoRAG** wins on multi-hop chains where the answer requires linking two facts together.
- **Mem0** wins on personalization: "remember that I prefer terse one-paragraph answers".

## Costs

Triple extraction is the biggest line item — one LLM call per chunk. Cache it aggressively. Once the graph is built, queries are fast (graph algorithms are seconds, not minutes, even at hundreds of thousands of nodes).

## When the graph is wrong

If your corpus is many independent documents that never reference each other, a graph adds cost without lift. Use graph retrieval when the corpus *is itself* a network — encyclopedias, scientific literature, legal cases, transcripts of conversations between people.
