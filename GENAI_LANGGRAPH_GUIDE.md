# LangGraph Guide

## What LangGraph Is
LangGraph models a workflow as nodes operating on shared state connected by edges. In this project it makes the RAG sequence explicit and gives each stage a status/error boundary.

## LangChain Versus LangGraph
LangChain provides documents, prompts, model integrations, embeddings, and vector stores. LangGraph provides the stateful execution graph that orders validation, retrieval, context construction, generation, and source preparation.

## State
`RAGState` is a `TypedDict` with optional fields for `question`, `top_k`, `documents`, `context`, `answer`, `sources`, `retrieval_scores`, `error`, and `status`. Each node receives and returns a state dictionary.

## Nodes and Flow
```text
START
  |
validate_node -> retrieve_node -> context_node -> generate_node -> source_node -> END
```

- `validate_node` calls `validate_question`, strips valid questions, and normalizes top-k. Invalid input receives `validation_failed`, an answer message, and empty result fields.
- `retrieve_node` calls `retrieve_documents_with_scores`, stores documents and scores, and converts exceptions through `sanitize_error_message`.
- `context_node` calls `has_retrieved_context` and `build_context`. Missing usable content produces the fallback answer and `no_context`.
- `generate_node` calls `generate_answer` and records `generated`; model failures become sanitized `generation_failed` state.
- `source_node` maps documents through `build_safe_source` for restricted frontend metadata.

## Graph Construction and Execution
`build_rag_graph` creates a `StateGraph(RAGState)`, registers five named nodes, adds linear edges, and compiles it. `run_rag_graph` builds the graph, creates initial state with default/received top-k and empty accumulators, then invokes it.

## Reliability-Aware Behavior
Validation failures stop meaningful downstream work because later nodes return the existing state for the failed status. Retrieval errors retain a safe error and empty documents. Context failures avoid model invocation. Generation errors expose only a generic safe message. The UI also catches unexpected exceptions around graph execution.

## Important Nuance
The graph is linear and has no conditional edge or agent loop. It does not implement hypothetical tools, planning agents, or autonomous retries. Its value here is explicit state and ordered reliability handling.

## Interview/Viva Questions
**Why LangGraph here?** To represent the RAG stages as inspectable state transitions with status and error fields.

**What is the state?** The `RAGState` dictionary carrying question, retrieved material, answer, sources, scores, and status.

**What happens for an empty question?** `validate_node` returns `validation_failed` and the safe empty-question message.

**What happens with no usable context?** `context_node` sets the fallback answer and `no_context`; generation is skipped.

**Is this an agent graph?** No. It is a fixed RAG workflow with five application nodes.
