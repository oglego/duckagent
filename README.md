# duckagent

🚧 WORK IN PROGRESS 🚧

A DuckDB extension that brings local, LLM-powered per-row data enrichment to SQL — no API keys, no cloud calls, no data leaving your machine. Powered by [agent.cpp](https://github.com/mozilla-ai/agent.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp) running a GGUF model of your choice.

It implements `ai_enrich`, a scalar function in the spirit of Databricks' [`ai_enrich`](https://docs.databricks.com/) — given a row of content and a schema of fields to fill in, it runs a local model to generate those fields, grammar-constrained to valid structured output.

```sql
SELECT ai_enrich(
    'Anthropic is an AI safety company based in San Francisco.',
    '["industry","headquarters_city"]'
);
```
```
┌────────────────────────────────────────────────────────────────────────────┐
│ struct(industry varchar, headquarters_city varchar)                        │
├────────────────────────────────────────────────────────────────────────────┤
│ {'industry': Artificial Intelligence, 'headquarters_city': San Francisco}  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Status

This is early, actively-developed v1. It works and produces real results, but the scope is deliberately narrow — see [Known limitations](#known-limitations) and [Roadmap](#roadmap) below before relying on it for anything serious.

## Quick start

### 1. Get a local model

You'll need a GGUF model file. Anything llama.cpp supports works; for `ai_enrich` specifically (short factual field generation, not open-ended chat), a small instruction-tuned model is a good starting point — e.g. Llama 3.2 1B Instruct or Gemma's edge-sized variants, quantized (`Q4_K_M` is a reasonable default). Larger/less-quantized models will generally be more accurate and slower; see [Performance & scaling](#performance--scaling).

### 2. Build

```sh
GEN=ninja make
```

This builds all of DuckDB and llama.cpp from source via CMake `FetchContent` — **the first build is slow** regardless of machine. Subsequent builds are incremental and much faster.

The build produces:
```
./build/release/duckdb
./build/release/extension/duckagent/duckagent.duckdb_extension
```

### 3. Run

```sh
export DUCKAGENT_MODEL_PATH=/path/to/your-model.gguf
./build/release/duckdb -unsigned
```

```sql
LOAD 'build/release/extension/duckagent/duckagent.duckdb_extension';

SELECT ai_enrich(
    'Anthropic is an AI safety company based in San Francisco.',
    '["industry","headquarters_city"]'
);
```

`DUCKAGENT_MODEL_PATH` is a temporary placeholder for model configuration until it's wired up as a proper DuckDB `SET` setting — see [Roadmap](#roadmap). The model is loaded lazily, once per process, the first time `ai_enrich` is called.

## `ai_enrich` reference

```sql
ai_enrich(content VARCHAR, schema VARCHAR) -> STRUCT(...)
```

- **`content`** — the row's content, as a `VARCHAR`. Build it however makes sense for your data, e.g. `name || ' is a company.'` or `to_json(row)`.
- **`schema`** — a JSON array of field names to generate, e.g. `'["industry","headquarters_city","year_founded"]'`. Must be a constant string (the return type is resolved from it at query bind time). Every field is generated as `VARCHAR` in this v1 (the "simple schema" form only — see [Known limitations](#known-limitations)).
- **Returns** — a `STRUCT` with one field per schema entry, in the order given.

### How output quality is protected

Grammar-constrained decoding (a GBNF grammar built per-query from your schema) guarantees the model's output is *syntactically* valid JSON matching your fields — but not that the *content* is sane. On top of that, `ai_enrich`:

- Caps each field value's length during generation, so a model can't ramble indefinitely into a field
- Runs a sanity check on each generated value (rejects empty values, stray structural characters, URLs, path-like fragments)
- Automatically retries a row once, at a higher sampling temperature, if any field fails that check — since a small model stuck in a bad deterministic decoding path can often escape it with a different sampling path
- Falls back to `NULL` for a field that's still bad after the retry, rather than displaying wrong-but-plausible-looking data

None of this can fix a model confidently stating an incorrect fact in a clean, well-formatted way — that's a genuine capability limit of small local models, not something a sanity filter can catch. Bigger/less-quantized models are more accurate; see below.

## Performance & scaling

`ai_enrich` runs a real local LLM generation per call — this is not a free-form SQL function, and won't feel like one. Expect somewhere in the neighborhood of **1–2+ seconds per row** depending on model size, quantization, and hardware. That makes it fine for small tables and ad-hoc lookups, but running it naively over hundreds of thousands of rows can take hours.

**If your enrichment key is low-cardinality** (a company name, a product category, anything with far fewer distinct values than rows), enrich the distinct values once and join back, rather than enriching every row:

```sql
CREATE TABLE company_enrichment AS
SELECT DISTINCT company_name,
       ai_enrich(company_name || ' is a company.',
                 '["industry","headquarters_city","year_founded"]') AS enrichment
FROM big_table;

SELECT t.*, e.enrichment.*
FROM big_table t
JOIN company_enrichment e USING (company_name);
```

This turns "N rows" into "N distinct values" — often a 50–200x reduction in real-world data. Materialize it as a table (not a view) so the LLM cost is paid once, and consider an incremental `INSERT ... WHERE NOT EXISTS` pattern to enrich only new distinct values over time rather than re-running from scratch.

If your data is genuinely high-cardinality on every column that matters (free text, unique IDs), this optimization doesn't apply — there's no way around one generation per row in that case with the current architecture.

`ai_enrich_distinct` packages the same pattern as a table macro. It returns one enrichment per distinct key; join its result back to the source table:

```sql
SELECT t.*, e.enrichment.*
FROM big_table AS t
JOIN ai_enrich_distinct(
    big_table,
    company_name,
    company_name || ' is a company.',
    '["industry","headquarters_city","year_founded"]'
) AS e ON t.company_name = e.ai_enrich_distinct_key;
```

`content_expr` may use any column from the source relation. Choose a key whose identical values should receive identical enrichment. Use `IS NOT DISTINCT FROM` instead of `=` in the join when the key can be `NULL` and those rows should be retained.

## Known limitations

This is v1, scoped deliberately narrow:

- **Simple schema only** — `schema` must be a flat JSON array of field names; every field comes back as `VARCHAR`. No typed/nested/enum schema (Databricks' "advanced schema" form) yet.
- **No grounding** — the model reasons only from `content` and its own training knowledge. No `knowledge_sources` (web search, vector index lookup) yet.
- **No `options` parameter** — no way to pass custom `instructions` or toggle rationale output yet.
- **`DUCKAGENT_MODEL_PATH` is an env var**, not a proper extension setting.
- **Single model path is process-wide** — the first model loaded via `DUCKAGENT_MODEL_PATH` is shared for the life of the process; there's no per-query model override.
- **Small local models can be confidently wrong** — no fact-checking or grounding exists yet to catch this class of error.

## Roadmap

Roughly in priority order:

- [x] `ai_enrich_distinct` — table macro for deduplicate-then-broadcast enrichment
- [ ] Per-context thread-count tuning, so DuckDB-level row parallelism scales cleanly instead of each model context competing for the whole machine
- [ ] `options` parameter (`instructions`, `enableRationale`)
- [ ] Typed/nested/enum schema (the "advanced schema" form)
- [ ] `DUCKAGENT_MODEL_PATH` → a real `SET duckagent_model_path` extension setting
- [ ] `knowledge_sources` (grounding via web search / local vector index)
- [ ] Pin the `agent.cpp` `FetchContent` dependency to a specific commit for reproducible builds (currently tracks `main`)

## Architecture notes

- Built on the [duckdb/extension-template](https://github.com/duckdb/extension-template) scaffold.
- Pulls in [agent.cpp](https://github.com/mozilla-ai/agent.cpp) via CMake `FetchContent`, which in turn pulls llama.cpp as its own dependency — both compiled from source as part of the extension build.
- `agent.cpp` is text-only; it has no multimodal/image input support (confirmed by inspecting its `Model` class directly), so this extension has no vision-related functionality and none is planned around it.
- Model weights are loaded once per process and shared across queries; each thread gets its own `Model` context (its own KV cache), consistent with DuckDB's per-thread scalar function execution model.

## Building

See [Quick start](#quick-start) above for the common path. A few additional notes:

- No VCPKG/external package dependencies are required — the template's original OpenSSL example dependency has been removed since `ai_enrich` doesn't need it.
- `make test` runs the SQL test suite under `./test/sql`.
- See `docs/UPDATING.md` for notes on keeping the DuckDB submodule in sync with upstream.
