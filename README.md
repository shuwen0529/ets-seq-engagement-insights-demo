# ets-seq-engagement-insights-demo

End-to-end demo: generate ETS-style engagement sequences, train baselines + a PyTorch LSTM sequence model, and extract behavioral insights using:
- cohort analytics (data-level patterns)
- counterfactual sensitivity tests (model-level "what-if" analysis)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline step-by-step (mirrors your 4-slide story)

```bash
PYTHONPATH=. python scripts/01_generate_data.py
PYTHONPATH=. python scripts/02_train_baselines.py
PYTHONPATH=. python scripts/03_train_lstm.py
PYTHONPATH=. python scripts/04_extract_insights.py
PYTHONPATH=. python scripts/05_make_slide4_summary.py
```

## Outputs

- Data:
  - `data/synthetic_users.csv`
  - `data/synthetic_events.csv`
- Metrics:
  - `outputs/metrics/split_users.json`
  - `outputs/metrics/baseline_metrics.json`
  - `outputs/metrics/lstm_metrics.json`
  - `outputs/metrics/insights_summary.json`
- Figures (drop directly into Slide 4):
  - `outputs/figures/fatigue_cohort.png`
  - `outputs/figures/time_gap_sensitivity.png`
- Slide-ready bullets:
  - `outputs/tables/slide4_summary.md`

## A disciplined approach

This project demo a disciplined approach:
- Establish strong baselines
- Quantify incremental lift
- Extract timing-sensitive insights
- Translate them into engagement strategy