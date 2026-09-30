# Ledgerline

Scores card transactions for fraud, turns messy bank descriptors into clean merchant categories, and serves both through a tested API, with every claim backed by a measured number.

> Work in progress. See [PLAN.md](PLAN.md) for the roadmap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Data

Download the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) data from Kaggle (accept the competition rules first) and put the CSVs in `data/raw/`. The `data/` folder is gitignored.

## Data findings

_Coming in Phase 1._

## Results

_Coming in Phase 3._
