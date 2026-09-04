# deepeval-test-eval-framework

An LLM evaluation framework built on [DeepEval](https://github.com/confident-ai/deepeval) and `pytest`. It contains four independent evaluation suites and a utility that turns the latest run into a styled, self-contained HTML report.

| Suite | What it evaluates | Metrics |
| --- | --- | --- |
| [`transcript-agent-eval`](transcript-agent-eval/) | A meeting-transcript summarizer that produces a summary **and** structured action items | `Summary Concision`, `Action Item Accuracy` (G-Eval) |
| [`RAG-agent-eval`](RAG-agent-eval/) | A FAISS-backed RAG question-answering agent | `Contextual Relevancy`, `Contextual Recall`, `Contextual Precision`, plus `Answer Correctness` & `Citation Accuracy` (G-Eval) |
| [`order-agent-eval`](order-agent-eval/) | A **single-turn tool-calling** customer-support agent answering order-status and refund questions | Agentic: `Task Completion`, `Tool Correctness`, `Prompt Alignment`, `Answer Relevancy`; Safety: `Bias`, `Toxicity`, `PII Leakage`; plus `Correctness` (G-Eval) |
| [`chat-agent-eval`](chat-agent-eval/) | A **multi-turn** ShopEasy support chatbot that keeps history across turns | `Turn Relevancy`, `Knowledge Retention`, `Conversation Completeness`, plus `Correctness` (Conversational G-Eval) |

Both the agents under test and DeepEval's LLM judges default to OpenAI models, but every model is configurable from `.env` — including non-OpenAI providers. See [Model configuration](#model-configuration).

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
- An OpenAI API key

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

That is the only **required** variable. Everything else — which model each agent and judge runs on — is optional and covered in [Model configuration](#model-configuration) below; sensible defaults apply if you set nothing.

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
ORDER_AGENT_MODEL = os.getenv("ORDER_AGENT_MODEL", "openai:gpt-4.1-mini")
```

Set the variable in `.env` and it wins; set nothing and the default applies. Cloning the repo and running with only an `OPENAI_API_KEY` works out of the box.

`pytest.ini` contains the two lines that make this importable from every suite:

```ini
[pytest]
pythonpath = .
```

Pytest normally only puts a test file's *own* folder on `sys.path`, so `from config import ...` would fail from inside `order-agent-eval/` without it.

### The variables

| Variable | Default | Format | Drives |
| --- | --- | --- | --- |
| `ORDER_AGENT_MODEL` | `openai:gpt-4.1-mini` | `provider:model` | the order agent |
| `ORDER_AGENT_JUDGE_MODEL` | `gpt-4.1-mini` | bare name | order-agent metrics |
| `RAG_AGENT_MODEL` | `openai:gpt-4.1-mini` | `provider:model` | the RAG generator |
| `RAG_EMBEDDING_MODEL` | `openai:text-embedding-3-small` | `provider:model` | the RAG embedder |
| `RAG_AGENT_JUDGE_MODEL` | `gpt-4.1-mini` | bare name | RAG metrics + `Synthesizer` |
| `SUMMARIZER_AGENT_MODEL` | `gpt-4.1-mini` | bare name | the summarizer **and** its G-Eval judge |
| `CHAT_AGENT_MODEL` | `gpt-4.1-mini` | bare name | the chatbot **and** its conversational judges |

### Two formats: the easy mistake

The format depends on **who reads the value**, not on which suite it belongs to:

| Consumer | Format | Example |
| --- | --- | --- |
| LangChain (`init_chat_model`, `init_embeddings`) | `provider:model` | `openai:gpt-4.1-mini` |
| OpenAI SDK and all DeepEval metrics | bare name | `gpt-4.1-mini` |

The `provider:` prefix is LangChain's way of choosing which integration class to build. The OpenAI SDK has no such concept and forwards the string as a literal model id — so passing `openai:gpt-4.1-mini` to a metric fails with:

```
openai.NotFoundError: 404 - The model `openai:gpt-4.1-mini` does not exist
```

If you see that, you've handed a `provider:model` value to something that wanted a bare name.

### Using a non-OpenAI provider

**LangChain-based agents** (`order_agent.py`, `rag_qa_agent.py`) accept any provider `init_chat_model` supports — just change the prefix:

```dotenv
ORDER_AGENT_MODEL=ollama:llama3.2
RAG_EMBEDDING_MODEL=ollama:nomic-embed-text
```

Each provider needs its integration package installed. `langchain-openai` and `langchain-ollama` ship in `requirements.txt`; others are one install away:

```powershell
pip install langchain-anthropic       # anthropic:claude-...
pip install langchain-google-genai    # google_genai:gemini-...
```

`init_chat_model` names the missing package if you skip this, so the failure is self-explaining.

**DeepEval judges** can be pointed at any provider DeepEval supports, via its own CLI — no code change:

```powershell
deepeval set-ollama --model llama3.2
deepeval set-anthropic --model claude-sonnet-4-5
deepeval set-litellm --model ...
```

Note that a metric constructed with an explicit `model=` argument ignores this global setting. Most metrics in this repo pass the suite's judge variable explicitly, which is deliberate — per-suite control — but it means the CLI switch above only affects metrics that *don't* specify a model.

### Known rough edge

`SUMMARIZER_AGENT_MODEL` and `CHAT_AGENT_MODEL` each drive both an agent *and* its judges, because those two agents call the OpenAI SDK directly and need bare names anyway. The order and RAG suites keep them separate (`*_AGENT_MODEL` vs `*_AGENT_JUDGE_MODEL`). Splitting the other two would make the scheme uniform.

---

## Running in CI

**Never commit `.env` or `.env.local`.** GitHub Actions does not read `.env` files — it sets real environment variables in the runner, and `config.py`'s `os.getenv` picks those up directly. `load_dotenv()` is a no-op when no file is present, so the same code works in both places.

Split variables by sensitivity in [`.github/workflows/eval-ci.yml`](.github/workflows/eval-ci.yml):

```yaml
env:
    # secret — stored in repo Settings → Secrets and variables → Actions
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    # not secret — plain values, visible and reviewable
    ORDER_AGENT_MODEL: "openai:gpt-4.1-mini"
    ORDER_AGENT_JUDGE_MODEL: "gpt-4.1-mini"

    # widen judge-call backoff; defaults (2 attempts, 5s cap) are too tight
    # to ride out a saturated OpenAI TPM window
    DEEPEVAL_RETRY_MAX_ATTEMPTS: "6"
    DEEPEVAL_RETRY_INITIAL_SECONDS: "2"
    DEEPEVAL_RETRY_CAP_SECONDS: "60"
```

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
- **`ModuleNotFoundError: No module named 'config'`** — `pytest.ini` is missing or lacks `pythonpath = .`. Pytest only puts a test file's own folder on `sys.path`, so the repo root has to be added explicitly.
- **`TypeError: unhashable type: 'ToolMessage'`** — DeepEval older than `4.0.9`. Run `pip install --upgrade deepeval`.
- **`429 ... rate_limit_exceeded` on TPM** — a *speed* limit, not a billing problem (that one reports `insufficient_quota`). Adding credit does not raise TPM; the limit comes from your usage tier. Throttle with `AsyncConfig` instead.
- **An evaluation fires during pytest collection** — a suite whose loop sits at module level runs on import. Keep evaluation code inside a `def test_...()` function, or invoke that file with `python`.
- **`FileNotFoundError` on a dataset** — confirm you're running from the repo root.
- **Authentication / 401 errors** — verify `OPENAI_API_KEY` is set in `.env` and the virtual environment is active.
