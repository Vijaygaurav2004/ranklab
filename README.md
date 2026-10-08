# RankLab

Rank LLM answers the way AI-training teams do: an AI judge scores every answer, you vote blind on
pairs, and RankLab turns both into leaderboards with confidence intervals. It also measures how far
you can trust the judge (position bias, agreement with you) and exports your votes as preference
data for RLHF-style training.

## Quick start (no API key)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ranklab demo                 # synthetic run with mock models and a mock judge
ranklab serve --run demo     # open http://127.0.0.1:8800
```

The demo's answers are fake ("mock quality: 0.82 ..."). Use it to try the app, not as results.

## A real run

```bash
cp .env.example .env         # add OPENAI_API_KEY (and optionally OpenRouter / Ollama)
ranklab models               # see which model names your key can use
ranklab generate --run oct-2026 --models openai:<model-a>,openai:<model-b>,ollama:<model-c>
ranklab judge    --run oct-2026 --judge openai:<strongest-model>
ranklab serve    --run oct-2026     # vote on every pair: 30 prompts x 3 model pairs = 90 votes
ranklab report   --run oct-2026     # writes runs/oct-2026/report.md
ranklab export   --run oct-2026     # writes chosen/rejected preference pairs (JSONL)
```

Every step resumes: rerun a command and it only does what is missing (failed calls are retried).
Any OpenAI-compatible endpoint works: `openai:`, `openrouter:` and `ollama:` (local models, no key).

## What it measures

| Output | How |
|---|---|
| Human ranking | Your blind pairwise votes. Answers are anonymised and sides shuffled; models are revealed only after you vote. |
| Judge ranking | The judge compares every pair twice, with the answers swapped. A verdict counts only if both orders agree. |
| Position bias | Share of pairs where the judge's verdict flips when the order is swapped, and whether it favoured the first or second slot. |
| Judge vs human | Agreement rate on pairs where both picked a winner, plus Cohen's kappa (agreement corrected for chance). |
| Rubric | The judge scores each answer 1–5 on correctness, instruction following, helpfulness and clarity. |
| Instruction checks | Word limits, bullet/line counts, JSON-only output, required or banned phrases, checked by code with no model in the loop. |
| Preference data | `{"prompt", "chosen", "rejected", ...}` JSONL, the format reward-model and DPO training use. Ties are left out. |

## How the ratings work

- **Bradley–Terry model.** Each model gets a strength *s*; the chance model *i* beats model *j* is
  *s_i / (s_i + s_j)*. Strengths are fitted from all battles with Hunter's MM algorithm and shown on an
  Elo-like scale (1000 = average, +400 = 10x the odds). Ties count as half a win each. One virtual tied
  game per pair keeps ratings finite when a model wins everything.
- **Confidence intervals.** The battles are resampled with replacement 300 times and refitted; the
  2.5th and 97.5th percentiles give the 95% interval. Overlapping intervals mean "not clearly different".
- **Why ask the judge twice.** LLM judges often prefer whichever answer they see first. Judging both
  orders and discarding flips removes that bias from the ranking, and the flip rate measures it.
- **Why compare with humans.** A judge is only useful if it agrees with careful human raters; kappa
  shows how much of that agreement is real rather than chance.

## The prompt set

`ranklab/data/prompts.jsonl` holds 30 prompts in six categories: reasoning, coding, writing, factual,
strict instructions and false premises (questions built on a myth, which a good answer corrects).
Use your own with `--prompts my.jsonl`; each line is
`{"id", "category", "prompt", "reference"?, "checks"?}`. Check types: `max_words`, `min_words`,
`bullet_count`, `line_count`, `json_object`, `contains_all`, `contains_any`, `contains_none`,
`ends_with`, `all_lowercase`, `regex`.

## Layout

```
ranklab/
  providers.py   OpenAI-compatible client with retries, plus a deterministic mock
  generate.py    collect answers concurrently, resumable
  judge.py       rubric and swapped-order pairwise judging
  metrics.py     Bradley-Terry, bootstrap CIs, Cohen's kappa
  report.py      summary, Markdown report, preference export
  server.py      FastAPI app for blind voting and the leaderboard
  static/        the web UI (plain HTML, CSS and JavaScript)
tests/           pytest suite, runs offline against the mock
```

Run the tests with `pytest`.
