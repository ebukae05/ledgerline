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

_Coming in Phase 3._
