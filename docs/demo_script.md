# Demo video script (2–3 minutes)

Goal: a reviewer who watches only this understands the problem, sees it work, and hears the two or three decisions that make it trustworthy.

## Before recording

```bash
docker compose up -d                     # API + Postgres (models trained, history seeded)
curl -s localhost:8000/health            # status "ok", merchant_llm true
python -m ledgerline.api.demo            # dry run once so the LLM cache is warm
```

- Terminal: large font (18pt+), dark theme, window about 120 columns wide.
- Browser tabs ready: the GitHub README (scrolled to the top) and the UI at `http://localhost:8000`.
- Record the screen plus your voice (Windows: Win+Alt+R with Xbox Game Bar, or OBS). Clear notifications first.

## Shot list

| Time | Show | Say (roughly; don't read it word for word) |
|---|---|---|
| 0:00–0:20 | README top: the three-row results table | "Ledgerline is a fraud-scoring and transaction-categorization service, the kind a card issuer runs. It catches 48% of fraud while bothering only 1% of good customers, versus 3% for hand-written rules, and every number here is reproducible." |
| 0:20–0:40 | Architecture diagram | "Training and serving share one feature implementation. The API pulls a card's history from Postgres and runs the exact functions used in training. A parity test proves the scores match to within one in a billion." |
| 0:40–1:30 | Web UI at `localhost:8000`, section 01: click Normal, Known fraud, Fraud the model misses, Broken request (or run `python -m ledgerline.api.demo` in a terminal) | Step 2: "A normal transaction from the test set, which the model never trained on, scores near zero." Step 3: "A real fraud gets flagged, and every flag comes with its top reasons, because a bank has to explain a decline." Step 4: "Bad input gets a clear 422 naming the field, never a crash." Step 5: "In a batch, one broken record is reported and the rest still score." |
| 1:30–1:50 | UI sections 02 and 03: categorize a few descriptors, then show the audit table filling in | "Raw bank descriptors become categories. Gemini knows DataCamp is education, which character patterns can't learn. And every decision, including model version and input hash, lands in an audit table, so we can answer 'why was this flagged?' months later." |
| 1:50–2:20 | README: fraud results table, then the merchant "split by merchant" bullet | "Two validation lessons. Splitting by time, the model is judged only on future transactions, and the test set was used exactly once. And for merchants, a random split scored 99.8% by memorizing names; splitting by merchant gave the honest 60%, which is why the LLM matters." |
| 2:20–2:45 | README: Performance table, then Limitations | "It serves at 29 milliseconds median, 63 requests a second per instance, with 139 tests and CI against a real Postgres. And here's what I'd do next: drift monitoring, since the threshold already drifted from 1% to 1.4% on future data, and shadow-mode rollouts for new models." |

## Tips

- Keep it under 3 minutes; cut a section rather than talk faster.
- If something fails on camera, re-record that section. Don't edit around a 500.
- Upload unlisted to YouTube or Loom and put the link at the top of the README.
