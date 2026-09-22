# Advanced RAG

A hands-on collection of **advanced Retrieval-Augmented Generation (RAG) architectures**, each implemented end-to-end in a standalone Jupyter notebook. Every technique tackles a specific weakness of naive "embed → retrieve → stuff-into-prompt" RAG — weak retrieval, hallucination, stale/incomplete context, or multi-hop reasoning — using a different strategy: self-critique, corrective retrieval, agentic routing, or a knowledge graph.

| # | Technique | Core Idea | Orchestration |
|---|-----------|-----------|----------------|
| 1 | [Self-RAG](#1-self-rag) | The model decides *when* to retrieve and grades its own answer for support & usefulness before returning it | LangGraph |
| 2 | [Corrective RAG (CRAG)](#2-corrective-rag-crag) | Retrieved documents are graded and corrected — refined, or discarded in favor of a live web search — before generation | LangGraph |
| 3 | [Agentic RAG](#3-agentic-rag) | A router agent chooses between a private knowledge base, live web search, or a direct answer, grading evidence at every hop | LangGraph |
| 4 | [Graph RAG (Neo4j)](#4-graph-rag-neo4j) | Facts are stored as an entity-relationship graph; questions are translated to Cypher for precise, multi-hop retrieval | LangChain + Neo4j |

---

## Why Advanced RAG?

Plain vector-similarity RAG has well-known failure modes:

- It retrieves even when it doesn't need to (e.g., "hi, how are you?").
- It trusts whatever the retriever returns, even when the chunks are irrelevant or contradictory.
- It has no fallback when the knowledge base simply doesn't contain the answer.
- It struggles with multi-hop, relational questions ("which actors worked with directors who also directed a 2010 sci-fi film?").

Each module in this repo is a self-contained experiment in fixing one of those problems.

---

## 1. Self-RAG

**Folder:** [`self-rag/`](./self-rag) · **Notebook:** [`self_rag1.ipynb`](./self-rag/self_rag1.ipynb)

Implements the Self-RAG pattern: the LLM is placed in the loop at every decision point, not just at generation time. It decides whether retrieval is even needed, checks whether the retrieved context is relevant, verifies that the generated answer is actually **supported** by the context (hallucination check), and rates whether the answer is **useful** enough to return — rewriting the question and retrying if not.

**Flow**

```
Question
   │
   ▼
decide_retrieval ──(no)──► generate_direct
   │ (yes)
   ▼
retrieve (FAISS similarity search)
   │
   ▼
is_relevant? ──(no)──► no_answer_found
   │ (yes)
   ▼
generate_from_context
   │
   ▼
is_sup? (is the answer supported by the retrieved context?)
   │
   ├──(fully supported)──► accept_answer
   └──(not supported)────► revise_answer ──► is_sup (loop)
                                                │
                                                ▼
                                        is_use? (is the answer actually useful?)
                                                │
                        ┌───────────────────────┼────────────────────────┐
                        ▼                        ▼                        ▼
                    return answer          rewrite_question           no_answer_found
                                            (loop back to retrieve)
```

- **Vector store:** FAISS, built from company documents (`documents/DSwithBappy_Company_Policies.pdf`, `_Company_Profile.pdf`, `_Product_and_Pricing.pdf`)
- **Embeddings:** `langchain-huggingface` (local, free)
- **LLM:** Groq (`langchain-groq`)
- **Decisions modeled as structured Pydantic outputs:** `RetrieveDecision`, `RelevanceDecision`, `IsSUPDecision`, `IsUSEDecision`, `RewriteDecision` — every routing decision is a typed, schema-validated LLM call instead of free-text parsing.

---

## 2. Corrective RAG (CRAG)

**Folder:** [`corrective rag/`](./corrective%20rag) · **Notebook:** [`corrective_rag.ipynb`](./corrective%20rag/corrective_rag.ipynb)

Implements Corrective RAG: instead of trusting the retriever blindly, every retrieved document is **scored** for relevance. Depending on the aggregate score, the pipeline takes one of three paths — refine and use what was retrieved, discard it and fall back to a rewritten web search, or treat the situation as ambiguous and blend both sources.

**Flow**

```
Question
   │
   ▼
retrieve_node (FAISS similarity search)
   │
   ▼
eval_each_doc_node (LLM grades each doc's relevance score)
   │
   ├── CORRECT (score > upper threshold) ─────► refine (decompose → filter → recompose) ─► generate
   ├── AMBIGUOUS (mixed scores) ───────────────► refine + rewrite_query → web_search ─────► ambiguous_node ─► generate
   └── INCORRECT (all scores < lower threshold)► rewrite_query_node → web_search_node ─────► generate
```

- **Refinement (strip-mine relevant knowledge):** long documents are decomposed into sentences, each sentence is kept/dropped by an LLM judge, then the surviving sentences are recomposed — so generation only sees the most relevant slivers of text, not entire noisy chunks.
- **Vector store:** FAISS, built from `documents/ML Book.pdf`
- **Web fallback:** Tavily Search API, triggered only when retrieval confidence is low
- **LLM:** Groq

---

## 3. Agentic RAG

**Folder:** [`Agentic_rag/`](./Agentic_rag) · **Notebook:** [`Agentic_RAG.ipynb`](./Agentic_rag/Agentic_RAG.ipynb)

An "industry-style" agent that doesn't just search a private knowledge base — it **routes**. A private KB is treated as the trusted, first-choice source; a live web search (Tavily) is used only as a fallback when KB evidence is graded as weak; and trivial questions get a direct answer with no retrieval at all.

**Flow**

```
Question
   │
   ▼
route_question ──► kb / direct
   │
   ├── direct ──────────────────────────────────────────────► direct_answer
   │
   └── kb ──► retrieve_kb ──► grade_kb_evidence
                                  │
                    ┌─────────────┴─────────────┐
                 strong                        weak
                    │                             │
                    ▼                             ▼
           generate_from_kb              tavily_web_search ──► grade_web_evidence
                                                                      │
                                                    ┌──────────────────┴──────────────────┐
                                                 strong                                  weak
                                                    │                                       │
                                                    ▼                                       ▼
                                          generate_from_web                     rewrite_query ──► retry retrieve_kb
                                                                                       │
                                                                          (still weak) ─► insufficient_evidence_answer
```

- **Vector store:** Pinecone (384-dim index, matched to the local embedding model)
- **Embeddings:** `sentence-transformers` via `langchain-huggingface` (free, local)
- **LLM:** Groq
- **Web fallback:** Tavily
- **Structured routing:** every branch (route / grade / rewrite) is a schema-validated LLM decision, not string matching
- **Source doc:** ingests the public LangGraph "Agentic RAG" docs page as its private KB for the demo

---

## 4. Graph RAG (Neo4j)

**Folder:** [`graph rag/`](./graph%20rag) · **Notebook:** [`graph_rag_neo4j.ipynb`](./graph%20rag/graph_rag_neo4j.ipynb)

Instead of chunk-and-embed, source documents are converted into a **knowledge graph** of entities and relationships (directors, movies, actors, genres, release years) using an `LLMGraphTransformer`, and stored in Neo4j. At query time, the question itself is translated into a Cypher query (`GraphCypherQAChain`), executed against the graph, and the retrieved facts are handed back to the LLM for the final answer — giving precise, multi-hop relational retrieval that plain vector search struggles with.

**Flow**

```
Documents
   │
   ▼
Chunking
   │
   ▼
LLM extracts entities + relationships  (LLMGraphTransformer)
   │
   ▼
Neo4j Knowledge Graph
   │
   ▼
User Question ──► Question → Cypher (LLM) ──► Neo4j retrieves connected facts
                                                        │
                                                        ▼
                                        Retrieved facts + question → LLM → Final Answer
```

- **Graph store:** Neo4j (via `langchain-neo4j`)
- **Extraction:** `langchain-experimental` `LLMGraphTransformer`
- **QA chain:** `GraphCypherQAChain` — reads the graph schema, generates Cypher, executes it, and synthesizes a natural-language answer
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Demo domain:** a small movie graph (directors → movies → actors → genres)

> ⚠️ **Before pushing this notebook publicly:** `graph_rag_neo4j.ipynb` currently has the Neo4j `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` **hardcoded in plaintext** in a code cell instead of loaded from `.env`. Rotate that database password and rewrite the cell to pull credentials from environment variables (see the pattern already used for `GROQ_API_KEY` two cells below it) before committing/pushing — otherwise the credentials will be visible in your GitHub history even if you fix the file later.

---

## Repository Structure

```
advanced rag/
├── Agentic_rag/
│   └── Agentic_RAG.ipynb
├── corrective rag/
│   ├── corrective_rag.ipynb
│   ├── documents/ML Book.pdf
│   ├── pyproject.toml
│   └── requirements.txt
├── graph rag/
│   └── graph_rag_neo4j.ipynb
├── self-rag/
│   ├── self_rag1.ipynb
│   ├── documents/*.pdf
│   ├── pyproject.toml
│   └── requirements.txt
├── requirements.txt          # shared deps for the root-level notebooks
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) |
| LLM inference | [Groq](https://groq.com/) (fast Llama / GPT-OSS inference) |
| Embeddings | `sentence-transformers` via `langchain-huggingface` (local, free) |
| Vector stores | [Pinecone](https://www.pinecone.io/) (Agentic RAG), [FAISS](https://github.com/facebookresearch/faiss) (Self-RAG, CRAG) |
| Graph store | [Neo4j](https://neo4j.com/) via `langchain-neo4j` + `langchain-experimental` |
| Web search | [Tavily](https://tavily.com/) |
| Structured outputs | [Pydantic](https://docs.pydantic.dev/) |
| Package/env management | [uv](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`) |

---

## Getting Started

Each sub-folder is a fairly independent notebook experiment. General setup:

```bash
# clone
git clone <this-repo-url>
cd "advanced rag"

# create an environment (pick one)
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# or, inside a specific technique's folder, using uv:
cd self-rag
uv sync
```

### Environment variables

Create a `.env` file (never commit it — already covered by `.gitignore`) with whichever of these are needed by the notebook you're running:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key       # Agentic RAG only
model=your_default_model_name

# Graph RAG only
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### Running a notebook

Open any of the four notebooks in Jupyter / VS Code and run the cells top to bottom — each one installs its own extra dependencies in the first cell(s) and is otherwise self-contained.

```bash
jupyter lab
```

---

## Choosing a Pattern

| If you need... | Use |
|---|---|
| Cheap, simple hallucination-resistance with self-checks | **Self-RAG** |
| Retrieval that recovers gracefully when the vector store returns junk | **Corrective RAG** |
| Multi-source routing (private KB vs. web vs. no retrieval at all) | **Agentic RAG** |
| Precise multi-hop / relational queries over structured facts | **Graph RAG** |

---

## License

Add a license of your choice (e.g., MIT) if you intend for others to reuse this code.
