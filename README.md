# Ledgerline

Scores card transactions for fraud, turns messy bank descriptors into clean merchant categories, and serves both through a tested API, with every claim backed by a measured number.

> Work in progress. See [PLAN.md](PLAN.md) for the roadmap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks]"
pre-commit install
pytest
```

## Data

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) data from Kaggle (accept the competition rules first) and put the CSVs in `data/raw/`. The `data/` folder is gitignored.

## Data findings

From [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb), on the 590,540 labeled training transactions.

1. **3.50% of transactions are fraud.** Always predicting "not fraud" scores 96.5% accuracy while catching nothing, so accuracy is not a usable metric here.
2. **The data spans 182 days, and fraud drifts week to week** (1.85% to 5.06%). The time-based split puts days 1–120 in train (413,378 rows), days 120–152 in validation (88,581) and days 152–182 in test (88,581). Each split ends up with 3.4–3.5% fraud.
3. **Volume doubles around days 20–25** (peak 6,852 transactions/day vs a median of 3,050), and fraud rate hits its lowest point in the same week. A model trained on that stretch sees an unusually "clean" period.
4. **Amount alone barely separates fraud.** Median fraud is $75.00 vs $68.50 for legit, and the largest transactions are all legit (max fraud $5,191 vs $31,937). The $5–20 band is the exception: 8.3% fraud (2.4× average), holding 10% of all fraud.
5. **Missingness is a signal.** Only 24% of transactions have an identity record, but those are 3.8× more likely to be fraud (7.9% vs 2.1%). 214 of 435 columns are more than half empty.
6. **Product type matters most among simple columns.** ProductCD `C` is 11.7% fraud vs 2.0% for `W` (5.8×). Mobile devices (10.2%) and credit cards (6.7% vs 2.4% for debit) are also elevated.
7. **Email domain varies:** outlook.com is 9.5% fraud vs 4.4% for gmail.com (among domains with 2,000+ transactions).
8. **Quiet hours are riskier.** The two lowest-volume hours of the (shifted) day run about 10% fraud vs about 2.3% in the busiest hours. The reference time is hidden, so these aren't known clock hours.

## Results

**Headline: at a 1% false-positive rate, the model catches 48% of fraud on held-out future data, vs 3% for hand-written rules and 11% for logistic regression.**

### Test set (days 152–182, evaluated once)

88,581 transactions the model never saw, all later in time than anything used for training or tuning. Evaluated a single time, after every choice was locked in; see [`reports/test_results.json`](reports/test_results.json).

| Model | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ top 1% |
|---|---|---|---|---|
| Random guessing | 0.035 | 0.500 | 1.0% | 3.5% |
| Rules (6 hand-written checks) | 0.096 | 0.717 | 2.6% | 17.4% |
| Logistic regression (9 features) | 0.143 | 0.749 | 11.3% | 31.3% |
| **LightGBM** (455 features, class-weighted) | **0.561** | **0.903** | **48.4%** | **89.5%** |

**At the deployed threshold** (picked on validation to flag at most 1% of legit transactions): on test it flagged 3.2% of transactions, caught **52% of fraud** and **45% of fraud dollars**, and 57% of flags were real fraud. It also wrongly flagged 1.44% of legit customers, above the 1% target. Score distributions drift over time, so a threshold set last month won't hold exactly this month; production would monitor and re-tune it.

Test is lower than validation (PR-AUC 0.561 vs 0.652). That's expected: validation chose the configuration, the tree count and the threshold, so it's slightly optimistic, and test is further in the future.

### How the model got there (validation, days 120–152)

Every LightGBM row is a mean over 2–3 random seeds; ± is the spread across seeds. Differences smaller than that spread are noise. Reproduce with `python -m ledgerline.experiments`; results in [`reports/experiments_val.json`](reports/experiments_val.json).

| Model | PR-AUC | Recall @ 1% FPR | Precision @ top 1% |
|---|---|---|---|
| Rules | 0.080 | 2.3% | 13.1% |
| Logistic regression | 0.172 | 12.8% | 34.9% |
| LightGBM, basic features (9) | 0.198 ± 0.003 | 15.7% | 40.6% |
| LightGBM + history features (25) | 0.238 ± 0.004 | 17.3% | 44.9% |
| LightGBM + dataset's own columns (439) | 0.641 ± 0.002 | 55.9% | 94.5% |
| LightGBM + history + dataset columns (455) | 0.643 ± 0.002 | 55.4% | 94.5% |
| …same, class-weighted (final model, 1 seed) | 0.652 | 57.3% | 94.7% |

- **History features** (per-card velocity, time since previous, amount vs card average, device/email seen before) add +0.040 PR-AUC over basic features, 10× the seed noise.
- **On top of the dataset's own columns they add nothing measurable** (+0.002, within noise). Those columns (C counts, D time gaps, V engineered features) already encode card history. History features stay in the model because they give human-readable reasons for a flag, which the anonymous V columns can't.
- **Class weighting** (`scale_pos_weight`) added +0.009 PR-AUC and +1.9 points of recall in a single run.

### Which history feature helped most (ablation)

Each group removed from the basic + history model, 3 seeds each:

| Removed | PR-AUC change |
|---|---|
| All history features | −0.040 |
| **Everything keyed on the card+address key** | **−0.022** |
| Time since previous transaction | −0.009 |
| Amount vs card average | −0.007 |
| Velocity counts (1h / 24h / 7d) | −0.007 |
| Card-level prior count | −0.006 |
| Card-level key (`card1` only) | −0.004 |
| Device/email seen before | −0.001 (noise) |

The card+address key is the most valuable single idea. LightGBM's own importance ranks `card_n_prior` first, yet removing it costs little: correlated features stand in for it. Importance shows what a model *used*; ablation shows what it *needed*.

## Merchant categorization

Raw descriptors like `[debit] PAYPAL *DATACAMP JYF7455M6J` → one of 17 categories. Developed on [DoDataThings/us-bank-transaction-categories-v2](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2) (68,000 synthetic rows from ~500 real merchant names, MIT).

**Headline: on merchants it had never seen, Gemini 3.1 Flash-Lite reached 92.3% accuracy at $0.02 per 1,000 descriptions, vs 60.0% for TF-IDF and 51.2% for hand-written rules.**

### Test set (10,157 descriptions, all from unseen merchants, evaluated once)

| Approach | Accuracy | Macro-F1 | Cost per 1k | Latency p50 |
|---|---|---|---|---|
| Rules (regex cleanup + 17 keyword rules) | 51.2% | 0.521 | $0 | 0.01 ms |
| TF-IDF char n-grams + logistic regression | 60.0% | 0.607 | $0 | 0.45 ms |
| **Gemini 3.1 Flash-Lite**, structured JSON output | **92.3%** | **0.927** | **$0.020** | 699 ms |
| Cascade: TF-IDF when ≥95% confident, else Gemini | 92.5% | 0.928 | $0.017 | — |

### Real-world check: my own bank transactions

65 transactions from my Bank of America account (Aug–Sep 2026), imported with `python -m ledgerline.merchants.import_bofa` (which replaces person names with `[NAME]`), after leaving out 19 cash-advance-app rows that fit none of the 17 categories. Labeled by me with an AI-drafted first pass; I reviewed every row. Only these aggregate numbers are published; the transactions stay in gitignored `data/private/`. Results in [`reports/merchants_real.json`](reports/merchants_real.json).

| Approach | Synthetic test | **My real transactions** |
|---|---|---|
| Rules | 51.2% | 53.8% |
| TF-IDF + LR | 60.0% | 61.5% |
| **Gemini 3.1 Flash-Lite** | **92.3%** | **86.2%** |
| Cascade | 92.5% | 83.1% |

- **Gemini loses ~6 points on real data, mostly to ambiguity, not ignorance.** 7 of its 9 errors are judgment calls: county and city payments (labeled Fees, predicted Utilities) and money received via Zelle or Apple Cash (labeled Transfer, predicted Income).
- **The cascade gets worse on real data.** TF-IDF's confidence was well calibrated on synthetic data but is confidently wrong on real statements, so the cascade keeps answers it should have passed on. Calibration learned on one distribution doesn't carry over to another.
- **Small and narrow:** 65 rows over ~29 distinct merchants, 8 of the 17 categories, many repeats (Uber rides, bus fares). Treat it as a sanity check, not a benchmark.

Results in [`reports/merchants_test.json`](reports/merchants_test.json); validation and tuning in [`reports/merchants_val.json`](reports/merchants_val.json). Reproduce with `python -m ledgerline.merchants.benchmark` (validation) then `--test` (once).

### What the numbers say

- **Split by merchant, or the score is fake.** A third of rows are exact duplicates and only ~500 merchants exist. With a random split, TF-IDF scores **99.8%** by memorizing merchant names. Split so every test merchant is unseen, the same model scores **60%**. Merchants are estimated from the text (first word specific to ≤2 categories), giving 2,142 groups and zero letters-only overlap between train and test.
- **The LLM wins because it knows merchants.** TF-IDF's mistakes are brands it's never seen: `BRILLIANT.ORG` and `DATACAMP` → Restaurants, `PRINCIPAL PMT` → Travel. No character pattern says DataCamp is education; world knowledge does.
- **The cascade barely pays here.** TF-IDF's confidence is well calibrated (≥90% confident: right 94% of the time; <50%: right 29%), so routing only unsure rows to the LLM works. But at $0.02 per 1,000, the LLM is already cheap, and matching its accuracy still sends 85% of rows to it. On validation, a 0.8 threshold sent 62% for 0.3 points less accuracy. The cascade matters when LLM cost or latency is the constraint, not here.
- **Remaining LLM errors are mostly label conventions**, not ignorance: Shopping ↔ Groceries (Costco, Walmart), Personal Care ↔ Shopping, Education ↔ Subscription (an online course is both).
- **Cost** is measured from token counts on batched calls (40 descriptions per request) at the Sep 2026 list price. Latency is a single-description request, i.e. what a live API call would see.

## How I validated this

- **Time-based split, no shuffling.** Train days 1–120, validation 120–152, test 152–182. A test proves every validation/test transaction is strictly later than every training one.
- **No feature sees the future.** History features only use transactions with an earlier timestamp for the same card. Tests delete all data after a cutoff and check that no earlier feature changes, and edit a later transaction and check that nothing earlier moves.
- **The card+address key.** The data has no customer ID. `card1` + `addr1` + first-use day (`day − D1`, since D1 counts days since the card's first transaction) stands in for one; only 3,112 of its 217,850 groups mix fraud and legit. Past fraud labels are never used as features.
- **Test set touched once.** `python -m ledgerline.evaluate` refuses to run a second time without `--force` and records the model version it scored.
- **Reproducible.** Pinned library versions, fixed seeds, deterministic LightGBM. Retraining gives a bit-identical model; the model version is a hash of its trees.

## Reproduce

```bash
python -m ledgerline.baselines     # rules + logistic regression, validation
python -m ledgerline.experiments   # all model variants + ablation (~1 hour)
python -m ledgerline.train         # final model -> artifacts/models/<version>/
python -m ledgerline.evaluate      # test set, once

python -m ledgerline.merchants.benchmark          # merchant approaches, validation
python -m ledgerline.merchants.benchmark --test   # synthetic test set, once
python -m ledgerline.merchants.benchmark --real   # your labeled transactions, once
```

The merchant LLM needs `GEMINI_API_KEY` in `.env` (see `.env.example`). Answers are cached in `data/processed/llm_cache/`, so re-runs cost nothing.
