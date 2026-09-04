# Early Telegram Popularity Analysis

> **Course:** Probability and Statistics (آمار و احتمال مهندسی)

An end-to-end probability and statistics project that asks: **how much can the first 30 minutes of a public Telegram post tell us about its performance after 24 hours?**

## What it does

- Collects public interaction snapshots at 5, 10, 20, 30, and 1,440 minutes
- Defines a channel-normalized popularity score from views, reactions, and forwards
- Performs descriptive analysis, robust summaries, and distribution diagnostics
- Measures early/final association and tests statistical hypotheses
- Uses conditional probability and Bayesian updating
- Trains a hand-built regularized logistic-regression model with uncertainty reporting

## Dataset

The included CSV contains public aggregate counters only—no message text, media, phone numbers, or private conversations. Re-running the committed snapshot retains 344 complete posts across 12 public channels after requiring all five checkpoints.

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analyze.py telegram_snapshots.csv
```

Generated charts and tables are written under `output/`.

## Optional data collection

Copy `telegram_config.example.py` to `telegram_config.py`, add your own Telegram API credentials locally, and run `collect_telegram.py`. The real config, session files, and SQLite database are ignored by Git.

## Results

The first substantial early/final association appeared by minute 5. On the committed snapshot, the manual logistic model reached 92.8% accuracy and an F1 score of 81.5% on the held-out set. These results are specific to the collected channels and period; collection completeness and Telegram's cumulative counters limit generalization.

This project was created collaboratively by Hossein Badr and Mohammad Jafari Naeemi.
