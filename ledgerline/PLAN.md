# Ledgerline: Project Plan

**One line:** A system that scores card transactions for fraud, turns messy bank descriptors into clean merchant categories, and serves both through a tested API, with every claim backed by a measured number.

**Who it's for:** Bank and card-issuer SWE / AI engineering roles (Amex, BNY, Capital One, FIS).

**Timeline:** 6 to 8 weeks, a few sessions per week.

---

## How to use this file with Claude Code

1. This file lives in the repo root as `PLAN.md`.
2. At the start of each phase, open Claude Code, switch to plan mode, and say: *"Read PLAN.md. We are starting Phase N. Help me plan it before writing code."*
3. Don't move to the next phase until you can answer that phase's **"Explain it"** questions without looking. Interviewers will ask exactly these.
4. After each phase, fill in the **Results log** at the bottom.

---

## Decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Fraud dataset | IEEE-CIS Fraud Detection (Kaggle) | 590,540 real e-commerce transactions, about 3.5% fraud, with a time column for leakage-safe splits |
| Merchant data | Real labeled data (see Phase 4 note: this turned into a hybrid) | Real-world credibility |
| Scope | Full plan, all phases | Depth over speed |
| Where to build | Your computer + Claude Code, pushed to GitHub | You drive, you learn |

**Still your call (decide before Phase 3):** which metric leads the README. Options: PR-AUC, recall at a fixed false-positive rate (e.g. 1%), or dollars of fraud caught per 1,000 flags. Think about which one a bank's fraud manager cares about, and why accuracy is useless here (predicting "not fraud" every time scores about 96.5%).

---

## Project rules (apply to every phase)

1. **No peeking at the future.** Every split is by time, and every feature is computed only from transactions that happened *before* the one being scored. Breaking this rule is called data leakage, and it produces results that look great and are fake.
2. **The oracle decides, not the model.** All scoring uses labels the model never saw during training. No LLM ever grades its own output.
3. **Baseline first.** Every "improvement" gets compared to something simpler. A number with no baseline means nothing.
4. **Tests from day one.** Every module gets tests when it's written, not at the end.
5. **Log every result.** Every experiment goes in the Results log with the date, what changed, and the metric.
6. **No secrets or personal data in git.** API keys go in `.env` (which is gitignored). Raw Kaggle data and any personal bank data never get committed.

---

## Production-ready requirements (apply across phases)

Someone else can run it, it doesn't break on bad input, and every number is reproducible.

**Must-have**

1. **Reproducible results** (Phases 2–3)
   - One command retrains the model and regenerates every metric: `python -m ledgerline.train` then `python -m ledgerline.evaluate`
   - Fixed random seeds and pinned library versions, so someone else gets the same numbers
   - The trained model is saved as a file with a version number attached
2. **The API never crashes on bad input** (Phase 5)
   - Missing fields, wrong types, negative amounts and unknown categories all return a clear 4xx, never a 500
   - In batch scoring, one bad record is logged and skipped while the rest still score
   - A test for each case
3. **Consistent features in training and serving** (Phases 3 and 5)
   - The exact same feature code runs in training and in the API (avoid training/serving skew)
   - A test that scores known transactions through the API and checks the scores match the offline pipeline
4. **Explainable decisions** (Phase 5)
   - Every flag returns its top reasons, e.g. "velocity: 6 transactions in 1 hour, 4x this card's normal"
5. **Audit trail** (Phase 5)
   - Every decision logged to Postgres with input, score, threshold, model version and timestamp
6. **Health and startup checks** (Phase 5)
   - `/health` reports whether the model is loaded and the database is reachable
   - If the model file is missing, the service refuses to start with a clear error
7. **Measured performance** (Phase 5)
   - p50/p95 latency from a load test, plus requests per second for one instance
8. **Runs anywhere** (Phases 0 and 5)
   - `docker compose up` on a fresh clone gives a working API (test it in a fresh folder)
   - CI runs lint and tests on every push; README setup steps actually work

**Nice-to-have**

- Threshold as config: change the flag threshold without retraining
- Drift monitoring: a script that compares this week's score distribution to training and warns on shift
- Shadow mode: a new model version scores alongside the old one without affecting decisions
- Deployed somewhere free, behind an API key

**Out of scope:** Kubernetes, microservices, Kafka/streaming. One well-tested API with an audit trail and real metrics beats a sprawling architecture.

---

## Repo layout (target)

```
ledgerline/
  PLAN.md
  README.md
  pyproject.toml          # dependencies
  .env.example            # key names only, no values
  .gitignore              # data/, .env, .venv/
  data/                   # gitignored: raw and processed data
  notebooks/              # exploration only, never imported
  src/ledgerline/
    data/                 # loading, time-based splitting
    features/             # leakage-safe feature builders
    models/               # baseline, logistic regression, LightGBM
    merchants/            # descriptor cleanup approaches
    eval/                 # metrics, reports
    api/                  # FastAPI app
  tests/
  .github/workflows/ci.yml
```

---

## Phase 0: Setup (1 to 2 sessions)

**Build**
- GitHub repo, Python virtual environment, `pyproject.toml`
- `pytest` running (even with one placeholder test)
- `ruff` for linting, pre-commit hook
- GitHub Actions running lint and tests on every push
- `.gitignore` covering `data/`, `.env`, `.venv/`

**Get the data (you do this part yourself)**
- Make a free Kaggle account, open the IEEE-CIS Fraud Detection competition page, and accept the competition rules. The download won't work until you accept them.
- Download and put the CSVs in `data/raw/`.
- Only `train_transaction.csv` and `train_identity.csv` have fraud labels. The competition's test files have **no labels**, so you can't evaluate on them. All your train/validation/test splits come from the train files.

**Checkpoint:** a push to GitHub triggers CI and it passes.

**Explain it**
- Why does CI matter on a solo project?
- Why should raw data never go in git?

---

## Phase 1: Understand the data (about 1 week)

**Build**
- A loader that reads the CSVs efficiently (they're large, so downcast number types or use Polars) and joins transactions with identity on `TransactionID`
- An exploration notebook covering: fraud rate, amount distribution for fraud vs non-fraud, missing values per column, and the time range
- `TransactionDT` is seconds from a hidden reference point, not a real date. Convert it to "day number" and plot fraud rate over time.
- A **time-based split** function: first ~70% of time for training, next ~15% for validation, last ~15% for test. Write a test that proves no test transaction happens before any training transaction.

**Checkpoint:** a short "Data findings" section in the README with 5 to 8 facts you discovered.

**Explain it**
- Why is a random split wrong for fraud data?
- What does 3.5% fraud mean for how you measure a model?

---

## Phase 2: Baselines (about 1 week)

**Build**
- **Rules baseline:** something a bank might hand-write, e.g. flag if amount is above a threshold, or the card is new, or the email domain is unusual. Tune the thresholds on validation only.
- **Logistic regression** on a small set of clean features
- An `eval` module that, given true labels and scores, reports: PR-AUC, ROC-AUC, recall at 1% false-positive rate, and precision at top-k flags
- A test for the eval module using a tiny hand-made example where you know the right answer
- Fixed seeds and pinned versions from here on (requirement #1)

**Checkpoint:** baseline numbers in the Results log. Everything after this gets compared to them.

**Explain it**
- What's the difference between ROC-AUC and PR-AUC, and why does PR-AUC matter more when fraud is rare?
- What does "recall at 1% FPR" mean in plain English for a bank?

---

## Phase 3: The real model (1 to 2 weeks)

**Build**
- LightGBM (or XGBoost) classifier
- **Leakage-safe features**, each computed only from earlier transactions:
  - per-card transaction count in the last 1 hour / 24 hours / 7 days (velocity)
  - this amount vs that card's average so far
  - time since that card's previous transaction
  - whether this card has used this device or email before
  - The dataset has no clean customer ID. Building a card/user key from card and address columns is a known trick. Research it, and write down how you built yours.
- Tests that prove a feature for transaction N never uses transaction N+1
- Feature code written so the API can import and reuse it unchanged (requirement #3)
- Handle class imbalance (try `scale_pos_weight` or leave it alone, and measure both)
- Pick the flag threshold on **validation**, then report on **test** exactly once
- `python -m ledgerline.train` / `python -m ledgerline.evaluate`, saving a versioned model file (requirement #1)

**Checkpoint:** a results table like this:

| Model | PR-AUC | Recall @ 1% FPR | Precision @ top 1% |
|---|---|---|---|
| Rules | | | |
| Logistic regression | | | |
| LightGBM, basic features | | | |
| LightGBM + velocity features | | | |

**Explain it**
- Which feature helped most, and how do you know? (feature importance plus an ablation: remove it and re-measure)
- Why do you only touch the test set once?

---

## Phase 4: Merchant descriptor cleanup (about 1 week)

**Important note from research:** public datasets of *real* raw bank descriptors with labels basically don't exist, because real transaction data is private. The public ones are synthetic. The strongest honest design is a hybrid:

- **Develop on a public synthetic set:** [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) on Hugging Face (68,000 rows, 17 categories, MIT license, built from 500+ real merchant names with realistic noise like `PP*SAFEWAY` or `AMAZON MKTPL*K8R2M5VN7`).
- **Test on real data you label yourself:** export 150 to 300 of your own bank or card transactions, label each one's category by hand, and use that as the **real-world test set**.
- **Privacy:** your personal transactions stay in `data/private/` (gitignored). Only aggregate accuracy numbers go in the README.

**Build**
Compare three approaches on the same test sets:
1. **Rules:** regex cleanup plus a keyword-to-category lookup
2. **Classic ML:** TF-IDF on character n-grams + logistic regression, trained on the synthetic set
3. **LLM:** Gemini with structured JSON output, forced to pick from the 17 categories

**Checkpoint:**

| Approach | Accuracy (synthetic test) | Accuracy (your real set) | Cost per 1k | Latency p50 |
|---|---|---|---|---|
| Rules | | | | |
| TF-IDF + LR | | | | |
| LLM | | | | |

**Explain it**
- Why might accuracy drop from synthetic to real data?
- When is the LLM worth its cost over the cheap model? Could you use the cheap model first and the LLM only when it's unsure?

---

## Phase 5: Serve it (1 to 2 weeks)

**Build**
- FastAPI app:
  - `POST /score`: transaction in; fraud score, flag decision, and top 3 reasons out
  - `POST /merchant`: raw descriptor in; category and confidence out
  - `GET /health`: model loaded + database reachable
- Refuse to start if the model file is missing (requirement #6)
- Pydantic validation on every input. Malformed records return a clear 4xx and never crash the service.
- A batch endpoint where one bad record is logged and skipped while the rest still process
- PostgreSQL table logging every decision (input hash, score, threshold, model version, time)
- Training/serving parity test (requirement #3)
- Docker Compose (API + Postgres)
- A load test (e.g. `locust` or a simple script) measuring p50/p95 latency and requests per second
- API tests, including bad-input tests

**Checkpoint:** latency numbers, total test count, and `docker compose up` works on a fresh clone.

**Explain it**
- Why log model version with every decision?
- What happens if the model file is missing when the API starts?

---

## Phase 6: Package it (a few days)

- README: problem, architecture diagram, results tables, "How I validated this", how to run it
- 2 to 3 minute demo video
- Write resume bullets from the Results log (real numbers only)
- Optional: deploy the API somewhere free, behind an API key

---

## Target resume bullets (fill in the X's with real numbers at the end)

- Trained a LightGBM fraud model on 590K+ card transactions using time-based splits and leakage-safe velocity features, achieving **X% recall at 1% FPR** vs **Y%** for a rules baseline
- Benchmarked rules, TF-IDF, and LLM approaches for classifying raw bank descriptors, reaching **X% accuracy on a hand-labeled real-world holdout** at **$Y per 1K** transactions
- Served real-time scoring via FastAPI + PostgreSQL audit logging at **Xms p50 latency**, with **N automated tests** and fault isolation for malformed records

---

## Week-by-week (flexible)

| Week | Phase |
|---|---|
| 1 | Phase 0 + start Phase 1 |
| 2 | Finish Phase 1, Phase 2 |
| 3 to 4 | Phase 3 |
| 5 | Phase 4 |
| 6 to 7 | Phase 5 |
| 8 | Phase 6 + buffer |

---

## Results log

| Date | Phase | What changed | Metric(s) | Notes |
|---|---|---|---|---|
| | | | | |
