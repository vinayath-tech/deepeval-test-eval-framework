# deepeval-test-eval-framework

An LLM evaluation framework built on [DeepEval](https://github.com/confident-ai/deepeval) and `pytest`. It contains four independent evaluation suites and a utility that turns the latest run into a styled, self-contained HTML report.

| Suite | What it evaluates | Metrics |
| --- | --- | --- |
| [`transcript-agent-eval`](transcript-agent-eval/) | A meeting-transcript summarizer that produces a summary **and** structured action items | `Summary Concision`, `Action Item Accuracy` (G-Eval) |
| [`RAG-agent-eval`](RAG-agent-eval/) | A FAISS-backed RAG question-answering agent | `Contextual Relevancy`, `Contextual Recall`, `Contextual Precision`, plus `Answer Correctness` & `Citation Accuracy` (G-Eval) |
| [`order-agent-eval`](order-agent-eval/) | A **single-turn tool-calling** customer-support agent answering order-status and refund questions | Agentic: `Task Completion`, `Tool Correctness`, `Prompt Alignment`, `Answer Relevancy`; Safety: `Bias`, `Toxicity`, `PII Leakage`; plus `Correctness` (G-Eval) |
| [`chat-agent-eval`](chat-agent-eval/) | A **multi-turn** ShopEasy support chatbot that keeps history across turns | `Turn Relevancy`, `Knowledge Retention`, `Conversation Completeness`, plus `Correctness` (Conversational G-Eval) |

The **agents under test run on local Ollama models**; DeepEval's **LLM judges run on OpenAI**. That split is a deliberate cost-optimisation — see [Cost optimisation: local agents, cloud judges](#cost-optimisation-local-agents-cloud-judges). Every model is configurable from `.env`, including switching either side back to a different provider. See [Model configuration](#model-configuration).

### Single-turn vs multi-turn

The last two suites look similar but exercise different halves of DeepEval, and the distinction drives everything else in this README:

| | `order-agent-eval` | `chat-agent-eval` |
| --- | --- | --- |
| Test case type | `Golden` → trace | `ConversationalTestCase` + `Turn` |
| Evaluation API | `dataset.evals_iterator()` | `evaluate()` |
| Metric family | single-turn (`SingleTurnParams`) | multi-turn (`MultiTurnParams`) |
| Custom judge | `GEval` | `ConversationalGEval` |
| Scores | one independent request at a time | the conversation as a whole |

Single-turn metrics will not accept a `ConversationalTestCase`, and vice versa — pick the metric family that matches the test case type.

---

## Repository layout

```
deepeval-test-eval-framework/
├── transcript-agent-eval/
│   ├── conftest.py            # MeetingSummarizer + prompt manager (agent under test)
│   ├── test_summary.py        # DeepEval suite for summary + action items
│   └── dataset/               # *.txt transcripts used as inputs
├── RAG-agent-eval/
│   ├── rag_qa_agent.py        # RAGAgent: chunk → embed → retrieve → generate
│   ├── test_rag.py            # DeepEval suite (synthetic goldens + retriever/generator metrics)
│   └── dataset/               # source documents for the RAG knowledge base
├── order-agent-eval/
│   ├── order_agent.py                      # LangChain tool-calling agent (agent under test)
│   ├── test_components_agentic_metrics.py  # agentic + safety metrics over traced runs
│   └── test_G_Eval_metrics.py              # G-Eval correctness against an expected_output
├── chat-agent-eval/
│   ├── chat_agent.py          # ShopEasy chatbot: manual tool loop, history preserved
│   └── test_chat_agent.py     # conversational metrics over a 4-turn dialogue
├── utils/
│   └── generate_report.py     # builds test_report.html from the latest run
├── config.py                  # central model configuration (reads .env)
├── pytest.ini                 # makes config.py importable from every suite
├── .deepeval/                 # DeepEval cache + latest run results (auto-generated)
├── requirements.txt
└── test_report.html           # generated report (output)
```

---

## Prerequisites

- Python 3.10+
- An OpenAI API key — used by the LLM judges
- [Ollama](https://ollama.com) running locally — used by the agents under test

Pull the two local models before running anything:

```powershell
ollama pull qwen2.5:3b                 # every agent under test
ollama pull nomic-embed-text-v2-moe    # RAG embeddings
```

Ollama serves on `http://localhost:11434` by default; override with `OLLAMA_BASE_URL`.

### 1. Create a virtual environment & install dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # PowerShell on Windows
pip install -r requirements.txt
```

> On macOS/Linux use `source venv/bin/activate` instead.

> **DeepEval must be `>= 4.0.9`.** Earlier releases crash with `TypeError: unhashable type: 'ToolMessage'` when `ToolCorrectnessMetric` scores tool calls captured from a LangChain agent. `requirements.txt` pins this.

### 2. Configure environment variables

Create a `.env` file in the repo root:

```dotenv
OPENAI_API_KEY=sk-...your-key...
```

That is the only required *secret*. A running Ollama with the two models pulled is the other hard requirement - see [Prerequisites](#prerequisites). Everything else — which model each agent and judge runs on — is optional and covered in [Model configuration](#model-configuration) below; sensible defaults apply if you set nothing.

`.env` is gitignored and must stay that way. In CI, set variables through the workflow instead — see [Running in CI](#running-in-ci). (A `CONFIDENT_API_KEY` is optional — see [Viewing traces](#3-viewing-traces-optional).)

### 3. Windows console encoding

DeepEval's progress output contains emoji. On a legacy Windows codepage this raises `UnicodeEncodeError: 'charmap' codec can't encode character` **during teardown**, which buries the real error under ~40 lines of `rich` internals. Set this before running:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## Model configuration

Every model this framework uses is declared in one place: [`config.py`](config.py). No model name is hardcoded inside a suite, so you can point the whole framework at different models — or a different provider entirely — **without editing Python**.

### How it works

`config.py` reads environment variables and supplies a default for each:

```python
ORDER_AGENT_MODEL = os.getenv("ORDER_AGENT_MODEL", "ollama:qwen2.5:3b")
```

Set the variable in `.env` and it wins; set nothing and the default applies. Cloning the repo, running Ollama, and supplying an `OPENAI_API_KEY` works out of the box.

> **`.env` wins over the defaults in `config.py`.** If you change a default in `config.py` but a stale `.env` still pins the old value, the `.env` value silently takes effect. Prefer leaving model pins out of `.env` entirely and treating `config.py` as the single source of truth.

`pytest.ini` contains the two lines that make this importable from every suite:

```ini
[pytest]
pythonpath = .
```

Pytest normally only puts a test file's *own* folder on `sys.path`, so `from config import ...` would fail from inside `order-agent-eval/` without it.

### The variables

| Variable | Default | Format | Drives |
| --- | --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL | every Ollama call |
| `ORDER_AGENT_MODEL` | `ollama:qwen2.5:3b` | `provider:model` | the order agent |
| `ORDER_AGENT_JUDGE_MODEL_OPENAI` | `gpt-4.1-mini` | bare name | order-agent metrics |
| `ORDER_AGENT_JUDGE_MODEL` | `qwen2.5:3b` | bare name | local judge object, available but unused |
| `SUMMARIZER_AGENT_MODEL` | `qwen2.5:3b` | bare name | the summarizer |
| `SUMMARIZER_JUDGE_MODEL_OPENAI` | `gpt-4.1-mini` | bare name | summarizer G-Eval judges |
| `SUMMARIZER_JUDGE_MODEL` | `qwen2.5:3b` | bare name | local judge object, available but unused |
| `CHAT_AGENT_MODEL` | `qwen2.5:3b` | bare name | the chatbot |
| `CHAT_JUDGE_MODEL_OPENAI` | `gpt-4.1-mini` | bare name | `Correctness` (Conversational G-Eval) |
| `CHAT_JUDGE_MODEL` | `qwen2.5:3b` | bare name | `Knowledge Retention` |
| `RAG_AGENT_MODEL` | `ollama:qwen2.5:3b` | `provider:model` | the RAG generator |
| `RAG_EMBEDDING_MODEL` | `ollama:nomic-embed-text-v2-moe` | `provider:model` | the RAG retriever's embedder |
| `RAG_SYNTH_EMBEDDER` | `nomic-embed-text-v2-moe` | bare name | `Synthesizer` context construction |
| `RAG_AGENT_JUDGE_MODEL_OPENAI` | `gpt-4.1-mini` | bare name | RAG metrics + `Synthesizer` critic |
| `RAG_AGENT_JUDGE_MODEL` | `qwen2.5:3b` | bare name | local judge object, available but unused |

Suites that offer both a local and a cloud judge expose them as `*_JUDGE_MODEL` (an `OllamaModel` object) and `*_JUDGE_MODEL_OPENAI` (a bare string). Swapping a suite between them is a one-word change at the metric's `model=` argument.

### Two formats: the easy mistake

The format depends on **who reads the value**, not on which suite it belongs to:

| Consumer | Format | Example |
| --- | --- | --- |
| LangChain (`init_chat_model`, `init_embeddings`) | `provider:model` | `ollama:qwen2.5:3b` |
| OpenAI SDK (incl. Ollama's `/v1` compat endpoint) | bare name | `qwen2.5:3b` |
| DeepEval metric, OpenAI judge | bare name | `gpt-4.1-mini` |
| DeepEval metric, Ollama judge | **object** | `OllamaModel(model="qwen2.5:3b")` |

`init_chat_model` splits on the **first** `:` only, so a tagged Ollama model survives: `ollama:qwen2.5:3b` parses as provider `ollama`, model `qwen2.5:3b`.

The `provider:` prefix is LangChain's way of choosing which integration class to build. The OpenAI SDK has no such concept and forwards the string as a literal model id — so passing `openai:gpt-4.1-mini` to a metric fails with:

```
openai.NotFoundError: 404 - The model `openai:gpt-4.1-mini` does not exist
```

If you see that, you've handed a `provider:model` value to something that wanted a bare name.

### Switching a judge between providers

A DeepEval metric resolves its `model=` argument through `initialize_model()`. A **bare string** falls through a chain of provider checks and, with no global provider flags set, lands on OpenAI:

```python
elif isinstance(model, str) or model is None:
    return OpenAIModel(model=model), True
```

Two consequences worth internalising:

- Passing `"qwen2.5:3b"` to a metric does **not** use Ollama. It asks OpenAI for a model of that name and fails with `openai.NotFoundError: 404 - The model 'qwen2.5:3b' does not exist`.
- To judge with Ollama you must pass an **object**: `OllamaModel(model="qwen2.5:3b", base_url=OLLAMA_BASE_URL)`. Putting an OpenAI model name *inside* that wrapper produces the mirror-image failure - `ollama._types.ResponseError: model 'gpt-4.1-mini' not found`.

Embeddings resolve through a separate function, `initialize_embedding_model()`, so the embedder's provider is **independent** of the judge's. That is what lets `RAG-agent-eval` run a `gpt-4.1-mini` critic against a local `nomic-embed-text-v2-moe` embedder inside the same `Synthesizer`.

DeepEval also offers global provider switches (`deepeval set-ollama`, `deepeval set-anthropic`, ...). This repo does **not** use them: every metric passes `model=` explicitly, which overrides the global setting anyway, and the global flags change how bare strings resolve in ways that are easy to forget.

**LangChain-based agents** (`order_agent.py`, `rag_qa_agent.py`) accept any provider `init_chat_model` supports - just change the prefix:

```dotenv
ORDER_AGENT_MODEL=openai:gpt-4.1-mini
RAG_EMBEDDING_MODEL=openai:text-embedding-3-small
```

Each provider needs its integration package installed. `langchain-openai` and `langchain-ollama` ship in `requirements.txt`; others are one install away:

```powershell
pip install langchain-anthropic       # anthropic:claude-...
pip install langchain-google-genai    # google_genai:gemini-...
```

---

## Cost optimisation: local agents, cloud judges

Every agent under test runs on a local Ollama model. Every metric that needs real judgement runs on `gpt-4.1-mini`. The asymmetry is the point.

### Why this split

The two roles have opposite cost profiles:

| | Agents under test | LLM judges |
| --- | --- | --- |
| Call volume | High - every turn, every golden, every retry | Low - a handful per test case |
| Task difficulty | Moderate - follow a prompt, call a tool | High - read a trace, apply a rubric, compare meanings |
| Cost if run on OpenAI | Dominates the bill | Marginal |

Moving the **agents** local captures nearly all of the saving. Moving the **judges** local would surrender measurement quality for very little of it.

### What we measured

Running `qwen2.5:3b` as a judge was tested and rejected on evidence:

- Given a *correct* answer, it scored 0.2 and justified that by quoting two semantically identical strings as a mismatch - it compares strings, not meanings. The same case scored 1.0 on a 5B model.
- `OllamaModel` exposes no token log-probs, so G-Eval falls back to the model's raw integer score. Scores become coarse (0.0 / 0.5 / 1.0) rather than a gradient, which makes thresholds tuned against OpenAI unreliable.
- A fully-local `RAG-agent-eval` run took **654s**. Ollama serialises requests on a single consumer GPU, so DeepEval's concurrent metric execution cannot overlap and wall time becomes the *sum* of every call. With OpenAI judges the same suite parallelises and finishes in single-digit minutes.

The agents themselves were fine: `qwen2.5:3b` handles the four-turn `chat-agent-eval` conversation with correct tool arguments throughout.

### Embeddings stay local

`RAG_EMBEDDING_MODEL` and `RAG_SYNTH_EMBEDDER` both use `nomic-embed-text-v2-moe`. Embeddings are the highest-volume, lowest-difficulty calls in the framework, so they are the best local candidates.

Keeping *both* on the same local embedder is also more correct than mixing. The synthesizer's embedder chooses which chunks become golden contexts; the agent's embedder performs the actual retrieval. Sharing one vector space keeps `Contextual Recall`/`Precision` measuring the retriever rather than a mismatch between two embedding spaces.

> `nomic-embed-text-v2-moe` has a **512-token** context, while `ContextConstructionConfig` defaults to `chunk_size=1024`. That would silently truncate every chunk, so `RAG-agent-eval` sets `chunk_size=400`.

### Prefer deterministic checks over judges

The cheapest judge is no judge. Where ground truth already exists in the repo, assert on it directly:

- `ToolCorrectnessMetric` in `order-agent-eval` takes no `model=` at all - it compares tool calls against `expected_tools`. It stays trustworthy no matter which judge is configured, and it is the first score to read when a run goes red.
- `chat-agent-eval` has `ORDERS` and `REFUND_POLICIES` in `chat_agent.py`. Checking a reply against those is a string assertion, not a judgement call.

Reserve G-Eval for genuinely fuzzy criteria - tone, coherence, "did it forget what the user said" - and give those a capable judge.

### Rough edges to know about

- **`.env` silently shadows `config.py`.** A stale pin in `.env` overrides any default you change in code, with no warning. Keep model pins out of `.env`.
- **Bare model strings default to OpenAI.** See [Switching a judge between providers](#switching-a-judge-between-providers) above.
- **The synthesizer swallows exceptions.** `generate_goldens_from_docs` wraps each document in a broad `except` and discards the traceback unless `DEEPEVAL_LOG_STACK_TRACES=1` is set. Without it, a failure surfaces only as `Generated 0 goldens` - and a suite that evaluates zero test cases **passes**. Set the variable, and assert the golden count is non-zero.
- **G-Eval loses log-prob scoring on Ollama**, as described above.

---

## Running in CI

**Never commit `.env` or `.env.local`.** GitHub Actions does not read `.env` files — it sets real environment variables in the runner, and `config.py`'s `os.getenv` picks those up directly. `load_dotenv()` is a no-op when no file is present, so the same code works in both places.

Split variables by sensitivity in [`.github/workflows/eval-ci.yml`](.github/workflows/eval-ci.yml):

```yaml
env:
    # secret — stored in repo Settings → Secrets and variables → Actions
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    # not secret - plain values, visible and reviewable.
    # Model pins can be omitted entirely; config.py's defaults already
    # select local agents + OpenAI judges.
    OLLAMA_BASE_URL: http://localhost:11434
    OLLAMA_KEEP_ALIVE: "30m"
    OLLAMA_NUM_PARALLEL: "1"

    # GitHub-hosted runners are CPU-only, so inference is several times slower
    # than on a local GPU and the 180s default task cap fails every judge call.
    # Set the per-attempt budget OR the per-task budget, not both - DeepEval
    # derives whichever one you leave unset.
    DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS: "600"
    DEEPEVAL_RETRY_MAX_ATTEMPTS: "3"
    DEEPEVAL_RETRY_INITIAL_SECONDS: "2"
    DEEPEVAL_RETRY_CAP_SECONDS: "60"

    # Without this the synthesizer discards tracebacks and a real failure
    # looks like "Generated 0 goldens" followed by a passing test.
    DEEPEVAL_LOG_STACK_TRACES: "1"
```

### Ollama on the runner

Because the agents run locally, CI has to provide Ollama itself. Install it, wait for the port, then pull the models - restoring the model cache *before* pulling so a warm cache makes the pulls no-ops:

```yaml
- name: Cache Ollama models
  uses: actions/cache@v4
  with:
    path: ~/.ollama/models
    key: ollama-qwen2.5-3b-nomic-v2moe-v1

- name: Install and start Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    (pgrep -x ollama >/dev/null || nohup ollama serve > ollama.log 2>&1 &)
    for i in $(seq 1 60); do
      curl -sf http://localhost:11434/api/tags >/dev/null && break
      sleep 2
    done

- name: Pull models
  run: |
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text-v2-moe

# Pay the cold start here rather than inside a timed judge call.
- name: Warm model
  run: |
    curl -s http://localhost:11434/api/generate \
      -d '{"model":"qwen2.5:3b","prompt":"hi","stream":false}' > /dev/null
```

A single shared cache key across all matrix jobs stores roughly 3GB once instead of four times.

`OPENAI_API_KEY` is still required - the judges use it. Both providers must be available in every job.

> Use `fail-fast: false` on the matrix. With four independent suites, `fail-fast: true` cancels the still-running jobs the moment any one of them fails, and a cancelled job produces no verdict at all - you cannot tell a passing suite from one that was killed mid-run.

A workflow-level `env:` block is inherited by every job and step, so it only needs declaring once. Model variables can be omitted entirely — `config.py`'s defaults apply — but listing them makes CI's configuration explicit rather than implied.

---

## Running the evaluations

Run all commands from the **repository root** so dataset paths resolve correctly.

Most files are pytest tests and run under `deepeval test run`. One is a plain script:

| File | Form | Command |
| --- | --- | --- |
| `transcript-agent-eval/test_summary.py` | pytest class | `deepeval test run <file>` |
| `RAG-agent-eval/test_rag.py` | pytest class | `deepeval test run <file>` |
| `order-agent-eval/test_components_agentic_metrics.py` | pytest function | `deepeval test run <file>` |
| `chat-agent-eval/test_chat_agent.py` | pytest function | `deepeval test run <file>` |
| `order-agent-eval/test_G_Eval_metrics.py` | **module-level script** | `python <file>` |

> **Don't point `deepeval test run` at the whole `order-agent-eval/` directory.** `test_G_Eval_metrics.py` runs its evaluation at module level, so pytest fires the whole evaluation just by *importing* it during collection. Name files individually, or run that one with `python`.

### `transcript-agent-eval`

Loads every `.txt` under [`transcript-agent-eval/dataset/`](transcript-agent-eval/dataset/), summarizes each transcript with the `MeetingSummarizer` agent, and scores the output against two G-Eval criteria (threshold `0.7`).

```powershell
deepeval test run transcript-agent-eval/test_summary.py
```

- Target a single test: `deepeval test run transcript-agent-eval/test_summary.py::TestSummary::test_eval_summarize`
- Useful flags: `-n <num>` for concurrency, `-c` to reuse DeepEval's cache, `-s` to show `print` output (pytest captures it on passing tests).

### `RAG-agent-eval`

Uses DeepEval's `Synthesizer` to generate goldens from the source document, runs them through the `RAGAgent` (retrieve + generate), and scores both the retriever and the generated answer.

```powershell
deepeval test run RAG-agent-eval/test_rag.py
```

> Goldens are regenerated by the LLM on every run, so the test inputs — and therefore the scores — differ between runs.

---

## `order-agent-eval` — single-turn tool-calling agent

[`order_agent.py`](order-agent-eval/order_agent.py) is a LangChain `create_agent` running `gpt-4.1` with two tools — `get_order_status` and `get_refund_policy` — over in-memory fixture data.

```powershell
deepeval test run order-agent-eval/test_components_agentic_metrics.py
python order-agent-eval\test_G_Eval_metrics.py
```

The agent can be run standalone to sanity-check it before evaluating:

```powershell
python order-agent-eval\order_agent.py
```

### Metrics

`test_components_agentic_metrics.py` scores two goldens — one order-status question, one refund question — against seven metrics:

| Metric | Threshold | What it checks |
| --- | --- | --- |
| `TaskCompletion` | 0.7 | did the agent actually accomplish what was asked? |
| `ToolCorrectness` | 0.5 | were the expected tools called? (deterministic, no LLM judge) |
| `PromptAlignment` | 0.5 | did the reply follow the system-prompt instructions? |
| `AnswerRelevancy` | 0.5 | is the answer on-topic for the question? |
| `Bias` | 0.5 | safety — biased language |
| `Toxicity` | 0.5 | safety — toxic language |
| `PIILeakage` | 0.5 | safety — leaked personal data |

`test_G_Eval_metrics.py` adds a single custom `GEval` judge, `Correctness` (threshold `0.8`), comparing the agent's answer against a hand-written `expected_output`.

### How the tracing is wired

Three pieces make the agent observable to DeepEval:

1. **`CallbackHandler`** is passed to `agent.invoke(..., config={"callbacks": [deepeval_callback]})`. This captures every LLM and tool call as a span, and populates `tools_called` on the trace automatically.
2. **`update_current_trace(output=response)`** records the agent's final answer as the trace output, which is what the trace-level metrics score.
3. **`@observe`** wraps the agent in each test file so the run produces a trace.

**Ground truth must be forwarded onto the trace.** This is the non-obvious part. `evals_iterator` scores metrics against the *trace*, and it reads `expected_tools` / `expected_output` from the trace — **not** from the `Golden`. Setting them on the `Golden` alone leaves the trace empty and the metric raises `MissingTestCaseParamsError`. The bridge is `get_current_golden()`:

```python
@observe(name="support agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_tools:
        update_current_trace(expected_tools=golden.expected_tools)

    return _support_agent(user_input)
```

### Controlling throughput

`AsyncConfig` governs how fast requests are issued. On a low OpenAI usage tier (30,000 TPM), the default — `run_async=True`, `max_concurrent=20` — will trip a `429 rate_limit_exceeded`. This suite runs serially:

```python
for golden in dataset.evals_iterator(
    metrics=[...],
    async_config=AsyncConfig(run_async=False),
):
```

Other options: `AsyncConfig(max_concurrent=2, throttle_value=1)` keeps async but caps simultaneous calls (`max_concurrent`) and spaces out their launches (`throttle_value`, in seconds). Pointing the judges at `gpt-4.1-mini` also helps, since it draws on a separate token pool from the agent.

---

## `chat-agent-eval` — multi-turn chatbot

[`chat_agent.py`](chat-agent-eval/chat_agent.py) is a ShopEasy customer-support chatbot on `gpt-4o`, calling the OpenAI API directly rather than through LangChain. It has the same two tools as the order agent, over its own fixture data (`ORD-1042`, `ORD-2099`, `ORD-7777` and four refund categories).

```powershell
deepeval test run chat-agent-eval/test_chat_agent.py
```

Or interactively, to try the bot by hand:

```powershell
python chat-agent-eval\chat_agent.py
```

### Why the tool loop is manual

`chat()` drives the tool-calling loop itself instead of delegating to LangGraph, and returns `(reply, history, tools_called)`. Keeping the full message history — including tool-call messages and tool results — is what makes multi-turn evaluation possible: each turn is replayed with everything that came before it, so metrics like `KnowledgeRetention` have something real to measure.

### The conversation under test

The suite drives a fixed four-turn dialogue:

```
1. "Give me status of ORD-1042?"
2. "What is the refund policy?"
3. "For clothing category?"
4. "What about food?"
```

Turns 3 and 4 are deliberately elliptical — neither repeats the word "refund". They only make sense if the bot carried context forward, which is exactly the failure mode this suite is designed to catch.

Each turn is appended as a `Turn(role=...)` pair and the whole dialogue becomes one `ConversationalTestCase`, scored by `evaluate()`.

### Metrics

| Metric | Threshold | What it checks |
| --- | --- | --- |
| `TurnRelevancy` | 0.7 | is each reply on-topic for the turn that prompted it? |
| `KnowledgeRetention` | 0.5 | does the bot remember facts established earlier in the conversation? |
| `ConversationCompleteness` | 0.5 | were the user's goals actually satisfied across the dialogue? |
| `ConversationalGEval` "Correctness" | 0.5 | custom judge — are the refund and order-status answers factually correct? |

The first three measure conversational *mechanics*; only `Correctness` looks at whether the content is true. That split is intentional — a bot can be relevant, retentive, and complete while stating a policy that does not exist.

> **Known limitation:** the `Correctness` judge currently receives only `ROLE` and `CONTENT`, and the `ConversationalTestCase` carries no `context` or `expected_outcome`. With no reference data, the judge grades *plausibility* rather than accuracy. To make it check real ground truth, pass the fixtures through `context=` and add `MultiTurnParams.CONTEXT` to `evaluation_params`.

---

## Analysing the results

### 1. The console table

Every run prints a per-test-case breakdown followed by aggregate metrics:

```
┌─────────────────────────────────────────────────────────────┐
│ ✅ test_case_0 (Passed 2 metrics)                           │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ Aggregate Metrics                                           │
│  Metric            │ Average Score │ Pass Rate │ Total      │
│ ───────────────────┼───────────────┼───────────┼─────────── │
│  Task Completion   │ 1.00          │ 100.00%   │ 2          │
│  Tool Correctness  │ 1.00          │ 100.00%   │ 2          │
└─────────────────────────────────────────────────────────────┘

✓ Evaluation completed 🎉! (time taken: 11.97s | token cost: 0.021804 USD)
```

Each metric carries a **score**, its **threshold**, and a **reason** — the judge's own explanation of the score. The reason is the most useful field when debugging a low score: it tells you whether the agent misbehaved or the criteria were ambiguous.

### 2. The HTML report

DeepEval persists the most recent run to `.deepeval/.latest_test_run.json` regardless of which suite produced it. [`utils/generate_report.py`](utils/generate_report.py) renders that into a standalone `test_report.html` with:

- a summary header (tests passed/failed, run duration, evaluation cost),
- a per-metric overview with average scores and pass/fail counts,
- a detailed card per test case showing each metric's score, threshold, reason, judge model, and cost.

```powershell
python utils\generate_report.py
start test_report.html        # Windows
```

> The report always reflects the **latest** run only, and overwrites the previous `test_report.html`. Run an evaluation first. Score colour coding: **green ≥ 0.70**, **amber ≥ 0.50**, **red < 0.50**.

#### Customising the report

`generate_report.py` builds the HTML as plain f-strings:

- **Styling:** edit the `<style>` block.
- **Score bands:** change the `>= 0.7` / `>= 0.5` cutoffs in the `score_class` logic.
- **Output location:** change `output_file = "test_report.html"`.
- **Content:** edit the metrics-overview and test-case loops.

### 3. Viewing traces (optional)

The span tree — which tool the agent chose, with what arguments, and how it recovered — is only visible in the Confident AI dashboard. There is no local trace viewer: without a key, DeepEval prints `No Confident AI API key found. Skipping trace posting.` and continues. Metric *scores* are computed locally either way.

```powershell
deepeval login          # opens a browser and stores the key
deepeval view           # opens the latest run in the dashboard
```

---

## Troubleshooting

- **`No results found at .deepeval/.latest_test_run.json`** — run an evaluation before generating the report.
- **`UnicodeEncodeError: 'charmap' codec can't encode character`** — Windows console encoding. Set `$env:PYTHONIOENCODING = "utf-8"`. This error appears *after* the real one, during `rich` teardown; scroll up to find the actual failure.
- **`MissingTestCaseParamsError: 'expected_tools' cannot be None`** — the golden's ground truth never reached the trace. Forward it with `update_current_trace(expected_tools=golden.expected_tools)` (see [How the tracing is wired](#how-the-tracing-is-wired)).
- **A single-turn metric rejects your test case** — single-turn metrics (`SingleTurnParams`) cannot score a `ConversationalTestCase`. Use the multi-turn equivalent: `ConversationalGEval` instead of `GEval`, and `MultiTurnParams` for `evaluation_params`.
- **`openai.NotFoundError: 404 - The model 'openai:gpt-4.1-mini' does not exist`** — a `provider:model` value was passed where a bare model name was expected. Metrics and the raw OpenAI SDK want `gpt-4.1-mini`; only LangChain's `init_chat_model` / `init_embeddings` understand the `openai:` prefix. See [Two formats](#two-formats-the-easy-mistake).
- **`ollama._types.ResponseError: model 'gpt-4.1-mini' not found (404)`** - an OpenAI model name was passed *inside* an `OllamaModel(...)` wrapper. The wrapper pins the call to Ollama regardless of the name. Use a bare string for an OpenAI judge, the object only for a local one.
- **`openai.NotFoundError: 404 - The model 'qwen2.5:3b' does not exist`** - the mirror image: a local model name was passed to a metric as a bare string, so it resolved to OpenAI. Wrap it in `OllamaModel(...)`.
- **`Generated 0 goldens` and the test still passes** - the synthesizer caught an exception and discarded the traceback. Re-run with `DEEPEVAL_LOG_STACK_TRACES=1` to see the real error, and assert the golden count is non-zero so an empty run cannot pass.
- **`MissingTestCaseParamsError: 'tools_called' cannot be None`** - the agent attempted its tools but every call failed validation, so DeepEval recorded none (`tools_called` is only appended in `on_tool_end`, not `on_tool_error`). Usually means the model emitted the tool's JSON *schema* as the arguments instead of the values - a small-model failure. Check the raw messages before blaming the metric.
- **`tenacity.RetryError[TimeoutError]` against Ollama** - judge calls exceeded the per-attempt budget. Ollama serialises requests, so concurrent metrics queue rather than overlap. Raise `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS`, cut the metric list, or move the judge to OpenAI.
- **A G-Eval judge scores 0.0 and complains that "conversation-level fields are missing"** - `evaluation_params` selects which fields are rendered into the judge prompt, and `ROLE`/`CONTENT` are deliberately skipped when building that block. Pass `SCENARIO` / `EXPECTED_OUTCOME` (and set them on the test case) so the block is populated; `ROLE` and `CONTENT` are appended automatically.
- **`ModuleNotFoundError: No module named 'config'`** — `pytest.ini` is missing or lacks `pythonpath = .`. Pytest only puts a test file's own folder on `sys.path`, so the repo root has to be added explicitly.
- **`TypeError: unhashable type: 'ToolMessage'`** — DeepEval older than `4.0.9`. Run `pip install --upgrade deepeval`.
- **`429 ... rate_limit_exceeded` on TPM** — a *speed* limit, not a billing problem (that one reports `insufficient_quota`). Adding credit does not raise TPM; the limit comes from your usage tier. Throttle with `AsyncConfig` instead.
- **An evaluation fires during pytest collection** — a suite whose loop sits at module level runs on import. Keep evaluation code inside a `def test_...()` function, or invoke that file with `python`.
- **`FileNotFoundError` on a dataset** — confirm you're running from the repo root.
- **Authentication / 401 errors** — verify `OPENAI_API_KEY` is set in `.env` and the virtual environment is active.
