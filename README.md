# deepeval-test-eval-framework

An LLM evaluation framework built on [DeepEval](https://github.com/confident-ai/deepeval) and `pytest`. It contains two independent evaluation suites and a utility that turns the latest run into a styled, self-contained HTML report.

| Suite | What it evaluates | Metrics |
| --- | --- | --- |
| [`transcript-agent-eval`](transcript-agent-eval/) | A meeting-transcript summarizer that produces a summary **and** structured action items | `Summary Concision`, `Action Item Accuracy` (G-Eval) |
| [`RAG-agent-eval`](RAG-agent-eval/) | A FAISS-backed RAG question-answering agent | `Contextual Relevancy`, `Contextual Recall`, `Contextual Precision`, plus `Answer Correctness` & `Citation Accuracy` (G-Eval) |

Both suites use OpenAI models — both to run the agent under test and as the LLM judge for DeepEval's metrics.

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
├── utils/
│   └── generate_report.py     # builds test_report.html from the latest run
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

### 2. Configure environment variables

Create a `.env` file in the repo root:

```dotenv
OPENAI_API_KEY=sk-...your-key...
```

The summarizer loads this automatically via `python-dotenv`. DeepEval also reads `OPENAI_API_KEY` for its judge models. (A `CONFIDENT_API_KEY` is optional — only needed if you want to push results to the Confident AI dashboard.)

---

## Running the evaluations

Run all commands from the **repository root** so the dataset paths resolve correctly. Use the `deepeval` CLI (`deepeval test run`) — it wraps `pytest` but adds DeepEval's evaluation lifecycle: a richer per-metric results table in the console and, importantly, it persists the run to `.deepeval/.latest_test_run.json`, which the report generator then reads.

### `transcript-agent-eval`

This suite loads every `.txt` file under [`transcript-agent-eval/dataset/`](transcript-agent-eval/dataset/), summarizes each transcript with the `MeetingSummarizer` agent, and scores the output against two G-Eval criteria (threshold `0.7`).

```powershell
deepeval test run transcript-agent-eval/test_summary.py
```

- To run a single metric/test, target it directly:
  ```powershell
  deepeval test run transcript-agent-eval/test_summary.py::TestSummary::test_eval_summarize
  ```
- Useful flags: `-n <num>` to run test cases concurrently, and `-c` to use DeepEval's cached results where available.

### `RAG-agent-eval`

This suite uses DeepEval's `Synthesizer` to generate goldens from the source document, runs them through the `RAGAgent` (retrieve + generate), and scores both the retriever and the generated answer.

```powershell
deepeval test run RAG-agent-eval/test_rag.py
```

> **Heads up — dataset path:** [`test_rag.py`](RAG-agent-eval/test_rag.py) currently references `./RAG-agent/dataset/theranos_legacy.txt`, but the file actually lives at [`RAG-agent-eval/dataset/theranos_legacy.txt`](RAG-agent-eval/dataset/theranos_legacy.txt). Update the two `document_paths` references in `generate_dataset()` and `build_test_case()` to `./RAG-agent-eval/dataset/theranos_legacy.txt` before running, or the suite will fail to find the document.

---

## Generating the customised report

DeepEval persists the most recent run to `.deepeval/.latest_test_run.json`. The script at [`utils/generate_report.py`](utils/generate_report.py) reads that file and renders a styled, standalone `test_report.html` with:

- a summary header (tests passed/failed, run duration, evaluation cost),
- a per-metric overview with average scores and pass/fail counts,
- a detailed card per test case showing each metric's score, threshold, reason, judge model, and cost.

### Steps

1. **Run an evaluation first** with `deepeval test run` (see above) so a fresh `.deepeval/.latest_test_run.json` exists. The report always reflects the *latest* run.

2. **Generate the report** from the repo root:

   ```powershell
   python utils/generate_report.py
   ```

   You should see:

   ```
   ✅ Report generated: test_report.html
   📂 Open it in your browser to view the results
   ```

3. **Open the report** in a browser:

   ```powershell
   start test_report.html        # Windows
   # open test_report.html       # macOS
   # xdg-open test_report.html    # Linux
   ```

> The report is written to `test_report.html` in the current working directory and overwrites any previous report. Score colour coding: **green ≥ 0.70**, **amber ≥ 0.50**, **red < 0.50**.

### Customising the report

`generate_report.py` builds the HTML as plain f-strings, so it's easy to tweak:

- **Styling:** edit the `<style>` block (colours, layout, the `score-good` / `score-warning` / `score-bad` thresholds).
- **Score bands:** change the `>= 0.7` / `>= 0.5` cutoffs in the `score_class` logic.
- **Output location / name:** change the `output_file = "test_report.html"` line.
- **Content:** add or remove sections by editing the metrics-overview and test-case loops.

---

## Troubleshooting

- **`No results found at .deepeval/.latest_test_run.json`** — run a `deepeval test run` evaluation before generating the report.
- **`FileNotFoundError` on a dataset** — confirm you're running from the repo root, and check the RAG dataset path note above.
- **Authentication / 401 errors** — verify `OPENAI_API_KEY` is set in `.env` and the virtual environment is active.
