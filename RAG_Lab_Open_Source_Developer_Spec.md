# RAG Lab: Open-Source E-commerce Retrieval-Augmented Generation Demo

**Developer specification · RAG-first prototype · October 2026**

## 1. Mission

Build a local-first, evaluation-driven Retrieval-Augmented Generation
(RAG) demo using synthetic e-commerce data. RAG is the product, not a
small feature inside a broader SaaS platform.

The demo should prove that the system can: - Answer product questions
from retrieved evidence rather than model memory. - Handle both semantic
questions and exact terms such as SKUs, brands, model numbers, and
attributes. - Respect structured constraints such as price, stock,
category, brand, and compatibility. - Compare products and explain
recommendations with citations to source records. - Recognize when the
catalog does not contain enough evidence and abstain rather than invent
facts. - Make retrieval inspectable and measurable. - Compare retrieval
strategies with a repeatable evaluation set. - Run locally without paid
APIs or proprietary hosted services.

**Core principle:** an e-commerce catalog is partly unstructured text
and partly structured data. Do not embed every field into one blob and
pretend vector similarity is enough. Use structured filtering for hard
constraints and RAG for semantic matching, evidence gathering, and
grounded explanations.

## 2. What we are building

Call the prototype **RAG Lab**. It is an experimental workbench, not a
customer-facing SaaS product.

The app has four main areas: 1. **Ask the Catalog** --- ask
natural-language questions and receive a grounded answer with citations.
2. **Retrieval Inspector** --- inspect the query, filters, candidate
results, scores, reranking, and final context. 3. **Evaluation Lab** ---
run a benchmark over synthetic queries and compare pipeline
configurations. 4. **Dataset Explorer** --- inspect products,
relationships, source records, and generated evaluation examples.

Use Streamlit for this small interface. Grafana is optional for
infrastructure monitoring; it is not the main UI. Do not build a
separate React frontend, billing, multi-tenant accounts, store
connectors, or a production SaaS dashboard for this demo.

## 3. Recommended open-source stack

  ------------------------------------------------------------------------
  Layer                   Recommendation           Purpose
  ----------------------- ------------------------ -----------------------
  Language                Python 3.11+             Main application and
                                                   RAG pipeline

  UI                      Streamlit                Fast interactive demo
                                                   and retrieval debugger

  Vector / hybrid         Qdrant, local Docker     Dense vectors, sparse
  retrieval               deployment               vectors, metadata
                                                   payloads, hybrid query
                                                   stages

  Lexical baseline        BM25 via `rank-bm25` or  Exact terms, SKUs,
                          a tested Qdrant          brand and model names
                          sparse-retrieval path    

  Embeddings              `Qwen3-Embedding-0.6B`   Local multilingual
                                                   dense embeddings;
                                                   verify the model
                                                   card/license before
                                                   distribution

  Reranker                `Qwen3-Reranker-0.6B`    Cross-encoder reranking
                                                   of a shortlist; verify
                                                   model card/license

  Local generation        Qwen3 instruct model     Local LLM inference
                          through Ollama           without a hosted API
                                                   key; select a model
                                                   size that fits
                                                   available hardware and
                                                   verify its license

  Data models             Pydantic                 Typed products, query
                                                   plans, retrieved
                                                   evidence, and answer
                                                   schemas

  Data processing         pandas + standard Python Synthetic catalog
                                                   generation and
                                                   evaluation tables

  Evaluation              Custom gold-set          Measure retrieval and
                          metrics + Ragas as an    answer quality
                          optional secondary       independently
                          evaluator                

  Testing                 pytest                   Unit, integration, and
                                                   regression tests

  Experiment tracking     JSONL/CSV initially;     Record configuration
                          optionally MLflow        and benchmark results

  Local services          Docker Compose           Reproducible Qdrant
                                                   setup; Ollama can run
                                                   on the host or in a
                                                   container

  Optional observability  Prometheus + Grafana +   Only if time permits;
                          OpenTelemetry            not required for the
                                                   core demo
  ------------------------------------------------------------------------

### License and cost policy

-   No paid APIs, API keys, hosted embedding endpoints, or required
    cloud services.
-   All application code should use open-source dependencies with
    compatible licenses.
-   Prefer model weights released under a permissive license such as
    Apache-2.0, but check each exact model repository and revision.
    "Open weights" does not automatically mean OSI-approved open source.
-   Record dependency and model licenses in `THIRD_PARTY_NOTICES.md`.
-   Download model weights once and cache them locally. The first
    download still requires internet access unless the models are
    preloaded.
-   "Free" means no per-request service fee; hardware, electricity,
    disk, and setup time are still costs.

### Why this stack?

Qdrant supports multi-stage hybrid queries that combine dense and sparse
retrieval and fuse candidate lists, including with Reciprocal Rank
Fusion (RRF). Qwen3 embedding and reranking models are designed for
retrieval/reranking and advertise multilingual coverage. Ollama supports
local model inference, while Streamlit is an open-source Python
framework for quickly building data apps. These are starting choices,
not a claim that one stack wins every benchmark. Measure on the actual
synthetic catalog.

## 4. Synthetic data: make it useful for evaluation

The synthetic data should resemble a real catalog, including difficult
and ambiguous cases. Do not generate random products without a coherent
schema or known answers.

### 4.1 Suggested dataset size

Start with: - 2,000--10,000 synthetic products. - 10--30 categories. -
50--200 brands. - 5--15 attributes per category. - Product variants
where relevant. - 500--1,000 benchmark queries, with gold relevant
product IDs and expected constraints. - A smaller hand-reviewed "demo
set" of 40--80 queries for live walkthroughs.

Begin with a smaller dataset and expand only after the pipeline works
end to end.

### 4.2 Product schema

Each product should contain fields such as:

``` json
{
  "product_id": "prod_000123",
  "sku": "AUDIO-X200-BLK",
  "title": "X200 Wireless Noise-Cancelling Headphones",
  "brand": "Northstar Audio",
  "category": "headphones",
  "description": "Over-ear wireless headphones with active noise cancellation...",
  "attributes": {
    "connection": "Bluetooth",
    "battery_hours": 38,
    "color": "black",
    "weight_g": 245,
    "noise_cancellation": true
  },
  "price": 129.99,
  "currency": "EUR",
  "in_stock": true,
  "rating": 4.4,
  "tags": ["wireless", "travel", "noise-cancelling"],
  "compatible_with": ["device_family_7"],
  "related_product_ids": ["prod_000124"],
  "source_record": "synthetic_catalog_v1",
  "updated_at": "2026-10-01T10:00:00Z"
}
```

Use numeric types for prices and measurements, booleans for stock and
features, and normalized IDs for relationships. Store the full canonical
product record in a local JSONL file or SQLite/PostgreSQL for
reproducibility; Qdrant is a retrieval index, not the only copy of the
data.

### 4.3 Include hard retrieval cases

Synthetic products should deliberately include: - Near-duplicate product
names. - SKUs and model numbers that differ by one character. - Products
with similar descriptions but different specs. - Exact brand/category
matches. - Missing attributes and incomplete descriptions. -
Out-of-stock products. - Products at just above/below price limits. -
Synonyms and paraphrases. - French and English queries; add Arabic if it
is relevant to the target demo. - Relationships such as accessories,
compatible devices, bundles, and alternatives. - Unanswerable questions
whose answer is absent from the catalog. - Conflicting-looking
descriptions that must be resolved using canonical structured fields.

### 4.4 Build the gold evaluation set from the synthetic generator

Because data is synthetic, exploit that advantage: generate both the
catalog and known answers together. Every benchmark query should have: -
`query_id` - `query_text` - `query_type` - `expected_product_ids` ranked
or labelled by relevance - `required_filters` - `expected_facts` -
`answerable` - `language` - `difficulty` - optional `reasoning_path` for
relationship questions

Example query types: 1. Exact lookup: "Find SKU AUDIO-X200-BLK." 2.
Attribute constraint: "Black headphones under €150 with at least 30
hours of battery." 3. Semantic need: "I need lightweight headphones for
long flights." 4. Comparison: "Compare the X200 and X210 on battery,
weight, and noise cancellation." 5. Relationship: "Which accessories are
compatible with device family 7?" 6. Multi-hop: "Find a compatible
accessory for the device that supports feature X and costs under €50."
7. Unanswerable: "Does this product include a five-year warranty?" when
no warranty data exists. 8. Ambiguous: "Show me the compact one" with no
prior context; the system should ask a clarification or state ambiguity.

Do not use an LLM judge as the only source of truth. The synthetic
generator can provide deterministic ground truth for product IDs,
filters, and expected attributes.

## 5. RAG pipeline

Implement the pipeline as explicit, testable stages.

``` mermaid
flowchart TD
    Q[User question] --> P[Query parsing and intent classification]
    P --> F[Validate structured constraints]
    P --> D[Dense query embedding]
    P --> S[Lexical / sparse query]
    F --> R1[Metadata-filtered dense retrieval]
    F --> R2[Lexical / sparse retrieval]
    R1 --> U[Union and deduplicate candidates]
    R2 --> U
    U --> RF[Rank fusion: RRF baseline]
    RF --> RR[Cross-encoder reranking]
    RR --> C[Evidence selection and context packing]
    C --> G[Local LLM with grounded-answer prompt]
    G --> V[Validate claims and citations]
    V --> A[Answer, citations, limitations]
    U --> I[Retrieval Inspector]
    RR --> I
    V --> I
```

### Stage 1 --- Query understanding

Classify the query into a small set of explicit intents: - exact
lookup - filtered product search - semantic discovery - comparison -
relationship / multi-hop question - unanswerable / clarification needed

Extract only supported structured constraints into a typed Pydantic
object. Example:

``` json
{
  "intent": "filtered_product_search",
  "keywords": ["headphones", "travel"],
  "filters": {
    "brand": null,
    "category": "headphones",
    "max_price": 150,
    "in_stock": true,
    "min_battery_hours": 30,
    "color": "black"
  },
  "needs_clarification": false
}
```

Do not let the LLM invent filters. Validate extracted fields against the
schema and catalog vocabulary. If local generation is too slow or
inaccurate, use deterministic parsing for exact IDs and simple filters,
then reserve the LLM for harder query interpretation.

### Stage 2 --- Structured filtering

Use canonical product fields for hard constraints: - price ceilings and
floors - in-stock requirement - category - brand - numeric
specifications - compatibility IDs

A vector similarity score is not a substitute for a hard condition. If
the user says "under €100," a €130 item must not appear as a valid match
just because it is semantically similar.

For this demo, metadata filters can live in Qdrant payloads; the
canonical catalog should remain separately available. Verify filter
behavior with tests.

### Stage 3 --- Hybrid candidate retrieval

Run two complementary retrievers: - **Dense retrieval:** semantic
similarity over a carefully constructed product text representation. -
**Lexical retrieval:** BM25 or sparse retrieval for exact terms, SKUs,
brand names, and technical attributes.

Fuse the lists using RRF as the baseline. Keep each retriever's
candidates and scores visible in the inspector. Tune weights only
against the evaluation set.

For product text, build a stable representation from title, brand,
category, description, and relevant attributes. Keep filterable values
as metadata too; do not rely on text alone for price or stock.

### Stage 4 --- Reranking

Retrieve a broad shortlist (for example, top 30--50 candidates across
retrievers), then rerank a smaller set with a local cross-encoder such
as `Qwen3-Reranker-0.6B`.

Compare: - dense only - lexical only - hybrid + RRF - hybrid + reranker

A reranker increases compute cost and latency. Keep it only if measured
relevance improvement justifies it. Cache embeddings and avoid reranking
the whole catalog.

### Stage 5 --- Context packing

Construct a compact context from the best evidence. Include: - product
ID and title - canonical facts relevant to the question - source record
ID - fields used to satisfy constraints - retrieval and reranker ranks
for debugging

Deduplicate repeated products. Avoid filling the context window with
near-duplicate descriptions. Use a maximum context budget and log
exactly what was passed to the generator.

### Stage 6 --- Grounded generation

Use a local instruct model through Ollama. The system prompt should
require the model to: - answer only from supplied evidence; - cite
product IDs for every product-specific claim; - distinguish catalog
facts from inference; - say when information is missing; - not infer
unsupported specifications, compatibility, warranty, or stock status; -
preserve exact prices and units from canonical records; - compare
products in a structured table when asked; - return "I cannot verify
this from the catalog" for unsupported questions.

Have the model return structured JSON with `answer`, `citations`,
`supported_facts`, `missing_information`, and `abstained`. Validate the
output with Pydantic. Render a readable answer only after validation.

### Stage 7 --- Citation and answer validation

Citations must point to retrieved product IDs/source records, not
invented URLs. Check that: - every cited ID exists in the selected
context; - cited products actually appeared in the evidence; - quoted
structured values match canonical fields; - the response does not
contain unsupported high-confidence facts.

For a demo, implement simple deterministic checks for prices, SKUs,
stock, and numeric attributes. Use an evaluator for semantic
faithfulness as an additional signal, not as a guarantee.

## 6. Optional advanced retrieval experiment: Graph RAG

Do **not** start with GraphRAG as the default. Add it as an experimental
retrieval mode only if your synthetic catalog has meaningful
relationships and your benchmark includes questions that need them.

Useful graph edges: - `COMPATIBLE_WITH` - `ACCESSORY_FOR` -
`ALTERNATIVE_TO` - `BELONGS_TO_CATEGORY` - `MADE_BY` - `BUNDLE_CONTAINS`

Represent relationships with stable product IDs and explicit provenance.
For a query like "Which accessory is compatible with the tablet that
supports protocol X and costs less than €500?", graph traversal can
identify candidate relationships while structured filtering validates
price and specifications. Retrieve textual evidence for the final
grounded answer.

Compare graph-assisted retrieval against hybrid RAG on the same query
set. Keep Graph RAG only if it improves multi-hop retrieval enough to
justify graph construction and maintenance. Recent benchmark work
specifically examines when graph-based RAG helps; graph complexity
should not be treated as a quality guarantee.

## 7. Evaluation is a core feature

The strongest part of the demo should be an evaluation lab, not a chat
window alone.

### 7.1 Retrieval metrics

Measure: - **Recall@K:** did the retriever include the known relevant
products? - **Precision@K:** how many retrieved products are relevant? -
**MRR:** how high is the first relevant result? - **NDCG@K:** does the
ranking put more relevant products near the top? - **Filter
correctness:** were hard constraints respected? - **Exact-ID success:**
does a SKU query return the correct product? - **No-result quality:**
does the system abstain or request clarification when appropriate?

### 7.2 Answer metrics

Measure: - **Faithfulness / groundedness:** are claims supported by
retrieved evidence? - **Answer correctness:** does the answer match
known synthetic facts? - **Citation precision:** do citations support
the claims they accompany? - **Abstention quality:** does the system
avoid answering unsupported questions? - **Completeness:** did it answer
the requested parts without inventing extra facts?

Ragas can provide supplementary metrics such as context precision,
context recall, response relevancy, and faithfulness. Keep deterministic
checks against generated gold labels as the primary reference wherever
possible.

### 7.3 Experiment matrix

  --------------------------------------------------------------------------
  Experiment     Retriever      Reranker       Generator      Purpose
  -------------- -------------- -------------- -------------- --------------
  A              BM25 only      No             Same local LLM Exact-term
                                                              baseline

  B              Dense only     No             Same local LLM Semantic
                                                              baseline

  C              Hybrid + RRF   No             Same local LLM Measure fusion
                                                              gain

  D              Hybrid + RRF   Yes            Same local LLM Measure
                                                              reranking gain

  E              Hybrid + RRF + Yes            Same local LLM Measure
                 filters                                      constraint
                                                              correctness

  F              Hybrid + graph Optional       Same local LLM Test multi-hop
                 expansion for                                benefit
                 selected                                     
                 intents                                      
  --------------------------------------------------------------------------

Keep the generator and prompt fixed when comparing retrievers.
Otherwise, changes cannot be attributed to the retrieval method.

### 7.4 Report the results

Save each run with: - Git commit or code version - dataset version -
embedding model and revision - reranker model and revision - generator
model and quantization - chunk/product representation version -
retrieval parameters - metric results - latency breakdown - memory use
if measured

Show a comparison table and examples of wins and failures. Do not claim
"state of the art" unless you actually reproduce a fair benchmark.

## 8. Streamlit demo interface

Keep the UI simple but technically revealing.

### Tab A --- Ask the Catalog

-   Query input and optional conversation follow-up.
-   Answer with product cards or a compact comparison table.
-   Citation chips showing product IDs and source fields.
-   Visible "not enough evidence" state.
-   Toggle for French / English queries, and Arabic if supported by the
    test set.

### Tab B --- Retrieval Inspector

Display: - parsed intent and filters; - dense candidates and similarity
scores; - lexical candidates and ranks; - fusion output; - reranker
scores; - final context passed to the LLM; - citations and validation
result; - per-stage latency.

This is the most important tab for demonstrating engineering judgment.

### Tab C --- Evaluation Lab

-   Select a benchmark subset or experiment configuration.
-   Run evaluation and show Recall@K, MRR, NDCG@K, filter correctness,
    faithfulness, and latency.
-   Compare experiments side by side.
-   Show failures and query-level diffs.
-   Export CSV/JSON results.

### Tab D --- Dataset Explorer

-   Search synthetic product records.
-   Inspect raw canonical fields and relationships.
-   Show which records support a benchmark query.
-   Identify unanswerable questions and missing fields.

Do not expose secrets or pretend a visualization is observability.
Grafana can be added later for service metrics if needed.

## 9. Suggested repository structure

``` text
rag-lab/
├── app.py
├── pages/
│   ├── 1_Ask_the_Catalog.py
│   ├── 2_Retrieval_Inspector.py
│   ├── 3_Evaluation_Lab.py
│   └── 4_Dataset_Explorer.py
├── src/
│   ├── config.py
│   ├── schemas/
│   │   ├── product.py
│   │   ├── query.py
│   │   └── answer.py
│   ├── data/
│   │   ├── synthetic_generator.py
│   │   ├── catalog_loader.py
│   │   └── benchmark_generator.py
│   ├── ingestion/
│   │   ├── normalize.py
│   │   ├── validate.py
│   │   └── index_builder.py
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── lexical.py
│   │   ├── hybrid.py
│   │   ├── filters.py
│   │   ├── fusion.py
│   │   └── rerank.py
│   ├── generation/
│   │   ├── ollama_client.py
│   │   ├── prompts.py
│   │   ├── answer_parser.py
│   │   └── citations.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── run_benchmark.py
│   │   └── reports.py
│   └── graph/
│       └── graph_retrieval.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── experiments/
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── THIRD_PARTY_NOTICES.md
└── README.md
```

Keep modules small and interfaces explicit. Do not build all the modules
at once; the roadmap below defines the order.

## 10. Implementation roadmap

### Phase 0 --- Hardware and model feasibility

-   Check available RAM, disk, CPU, and GPU/VRAM.
-   Select a generator model that can run locally on the target machine.
-   Verify exact model licenses and download requirements.
-   Test embedding, reranking, and generation latency on a small sample.
-   Write down the chosen model revisions and fallback configuration.

**Gate:** a local query can generate a response without any hosted API
or API key.

### Phase 1 --- Data and gold set

-   Implement deterministic synthetic catalog generation.
-   Include edge cases and explicit relationships.
-   Generate benchmark questions and gold product IDs/facts from the
    same source.
-   Validate data schema and uniqueness constraints.
-   Store versioned JSONL files.

**Gate:** each benchmark query has expected results or is explicitly
marked unanswerable.

### Phase 2 --- Baselines

-   Implement exact SKU lookup.
-   Implement BM25 retrieval.
-   Implement dense embedding retrieval.
-   Build a small command-line evaluation runner before building the UI.

**Gate:** baseline metrics are reproducible and saved to disk.

### Phase 3 --- Hybrid retrieval and filters

-   Add Qdrant.
-   Index product text and structured payload fields.
-   Implement dense + lexical candidate retrieval and RRF fusion.
-   Enforce hard filters for price, category, stock, brand, and specs.
-   Add deterministic tests for filter correctness and tenant isolation
    is not applicable to this single-user demo.

**Gate:** exact-ID and hard-filter tests pass; hybrid results can be
inspected.

### Phase 4 --- Reranking and context selection

-   Add a local cross-encoder reranker.
-   Compare hybrid results before and after reranking.
-   Implement context packing and deduplication.
-   Record stage timings and scores.

**Gate:** the benchmark quantifies whether reranking improves ranking
quality and at what latency cost.

### Phase 5 --- Grounded generation

-   Run the local generator through Ollama.
-   Define a strict evidence-only prompt.
-   Require structured output and validate it with Pydantic.
-   Implement citation validation and abstention behavior.
-   Add adversarial/unanswerable queries.

**Gate:** unsupported product claims are caught by tests, and citations
map to retrieved evidence.

### Phase 6 --- Streamlit workbench

-   Build the four tabs.
-   Add configuration selectors for retrieval modes.
-   Show retrieval trace, citations, failure cases, and evaluation
    comparison.
-   Keep the default flow usable in a live demo.

**Gate:** a reviewer can compare at least three retrieval configurations
without reading the source code.

### Phase 7 --- Advanced experiment and polish

-   Add graph-assisted retrieval only for relationship-heavy queries.
-   Add Ragas as a secondary evaluator.
-   Add caching, error handling, and performance profiling.
-   Write README instructions and license notices.
-   Record a short demo walkthrough using representative queries.

**Gate:** the final demo presents both successful and failed cases and
explains measured trade-offs.

## 11. Testing checklist

### Data and indexing

-   Duplicate IDs are rejected.
-   Required fields are validated.
-   Numeric attributes remain numeric.
-   Reindexing is repeatable.
-   Product updates remove stale indexed values.

### Retrieval

-   Exact SKU lookup returns the correct item.
-   Similar-looking SKUs do not cross-match silently.
-   Hard price and stock constraints are respected.
-   Dense and lexical results can be inspected independently.
-   Fusion deduplicates product IDs.
-   Reranker does not remove required metadata.

### Generation

-   Every citation references a retrieved product.
-   Prices and technical specs match canonical fields.
-   Unsupported warranty/compatibility claims are rejected.
-   Unanswerable questions lead to an explicit limitation.
-   Malformed model output fails safely.
-   The app handles model-server timeouts.

### Evaluation

-   A fixed benchmark gives reproducible results for deterministic
    stages.
-   Model-dependent metrics are versioned and their limitations
    documented.
-   Query-level failures can be exported.
-   Baseline and advanced configurations use the same benchmark.

## 12. Performance and resource strategy

Because this is a local demo, optimize for a good quality/latency
trade-off rather than production-scale throughput.

-   Cache product embeddings and rebuild only when the source text
    changes.
-   Persist the Qdrant collection between runs.
-   Batch embedding during ingestion.
-   Retrieve a manageable candidate pool before reranking.
-   Keep the reranker and generator configurable.
-   Run a small model first; larger models are optional experiments, not
    prerequisites.
-   Report stage timings separately: query parsing, embedding,
    retrieval, reranking, generation, and validation.
-   Avoid making hard latency claims until they are measured on the
    actual hardware.

## 13. What not to build in this demo

Out of scope: - Merchant authentication and multi-tenancy. -
Shopify/WooCommerce/PrestaShop integrations. - Billing, payments,
subscription plans, and usage metering. - Full SaaS administration. - A
separate frontend framework. - Fine-tuning as the first solution to
retrieval problems. - GraphRAG as the default for every query. -
Multi-agent orchestration without a measurable reason. - Hosted LLM,
embedding, reranking, or observability services. - A production-scale
distributed deployment.

The demo is meant to prove the RAG system itself.

## 14. Definition of done

The project is ready to present when: - It runs locally from documented
setup steps. - No paid API key or hosted service is required. - The
synthetic dataset and benchmark can be regenerated. - The app compares
dense, lexical, hybrid, and hybrid-plus-reranker retrieval. - Hard
structured constraints are tested. - Answers include validated citations
and can abstain. - Evaluation results and failure examples are
exportable. - Model and dependency licenses are documented. - A reviewer
can follow a query from parsing through retrieval and generation. - The
README explains limitations and reports actual benchmark results rather
than invented performance claims.

## 15. Live demo script

Use a short sequence that reveals why the design matters:

1.  **Exact SKU:** ask for a specific SKU and show lexical retrieval
    succeeding where semantic similarity alone may be unreliable.
2.  **Semantic need:** ask for a product suitable for a long flight and
    show dense retrieval contributing relevant candidates.
3.  **Hard constraints:** add a price ceiling and stock requirement;
    show invalid products being filtered out.
4.  **Reranking:** compare the hybrid candidate order before and after
    reranking.
5.  **Grounded comparison:** compare two products and click citations to
    verify each spec.
6.  **Unanswerable question:** ask about a warranty not present in the
    catalog and show the system abstaining.
7.  **Evaluation:** run the benchmark and show which approach wins for
    which query type.
8.  **Optional graph question:** ask a compatibility or accessory
    question and compare graph-assisted retrieval with the standard
    hybrid pipeline.

The strongest message is not "I used the newest framework." It is: **"I
built a measurable RAG system, designed retrieval around the data,
tested each stage against ground truth, and can show exactly where
quality improves or fails."**

## 16. Research references

The recommendations above are based on a combination of official
documentation and recent research reviewed on 13 October 2026. Re-check
model cards, project releases, and licenses before implementation
because they can change.

-   Qdrant, *Hybrid and Multi-Stage Queries*:
    https://qdrant.tech/documentation/search/hybrid-queries/
-   Qwen, *Qwen3 Embedding*:
    https://qwenlm.github.io/blog/qwen3-embedding/
-   Qwen3 Embedding repository:
    https://github.com/QwenLM/Qwen3-Embedding
-   Ollama documentation: https://github.com/ollama/ollama
-   Streamlit documentation: https://docs.streamlit.io/
-   Ragas metrics:
    https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
-   RAGFlow repository (useful as a reference implementation or
    alternative, not a requirement):
    https://github.com/infiniflow/ragflow
-   GraphRAG-Bench repository and research context:
    https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
-   RAGPerf, *An End-to-End Benchmarking Framework for
    Retrieval-Augmented Generation Systems*:
    https://arxiv.org/abs/2603.10765

### Final implementation principle

Start with a simple, measurable hybrid RAG baseline. Use the synthetic
dataset to determine whether reranking, query rewriting, or
graph-assisted retrieval adds value. Complexity is a hypothesis to test,
not a feature to add for its own sake.
