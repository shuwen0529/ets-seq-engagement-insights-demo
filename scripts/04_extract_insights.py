from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import Paths, load_cfg, ensure_dirs
from src.sequence_builder import build_user_sequences
from src.model_lstm import EngagementLSTM
from src.interpret import cohort_fatigue, counterfactual_add_recent_event, time_gap_sensitivity_sweep, plot_curve, _predict_batch


def main():
    paths = Paths()
    cfg = load_cfg(paths.config_path)
    ensure_dirs(paths)

    users = pd.read_csv(paths.data_dir / "synthetic_users.csv")
    events = pd.read_csv(paths.data_dir / "synthetic_events.csv")

    ckpt = torch.load(paths.outputs_dir / "lstm_model.pt", map_location="cpu")
    vocab = ckpt["vocab"]

    Xe, Xg, m, y, kept_users = build_user_sequences(
        events_df=events,
        users_df=users,
        vocab=vocab,
        max_len=cfg["sequence"]["max_seq_len"],
        min_len=cfg["sequence"]["min_seq_len"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EngagementLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # A) Cohort fatigue (data-level)
    fatigue_df = cohort_fatigue(events, users, window_days=14)
    fatigue_df.to_csv(paths.tables_dir / "fatigue_cohort_14d.csv", index=False)

    plt.figure()
    plt.bar(fatigue_df["events_last_window"].astype(str), fatigue_df["actual_rate"])
    plt.title("Actual outcome rate by events in last 14 days")
    plt.xlabel("events_last_14_days")
    plt.ylabel("actual_outcome_rate")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "fatigue_cohort.png", dpi=160)
    plt.close()

    # B) Counterfactual: add recent event
    add_ev = vocab.get("content_review", 1)
    base_pred = _predict_batch(model, Xe[:2000], Xg[:2000], m[:2000], device).mean()
    cf_pred = counterfactual_add_recent_event(model, Xe[:2000], Xg[:2000], m[:2000], add_ev, 2.0, device).mean()

    # C) Time-gap sensitivity sweep
    sweep = time_gap_sensitivity_sweep(model, Xe, Xg, m, device, gap_values=(1,3,7,14), n_samples=1200)
    sweep.to_csv(paths.tables_dir / "time_gap_sensitivity.csv", index=False)
    plot_curve(
        sweep, "gap_days_set_at_last_step", "avg_pred_prob",
        "Time Gap Sensitivity (avg predicted probability)",
        paths.figures_dir / "time_gap_sensitivity.png"
    )

    summary = {
        "counterfactual_add_recent_event": {
            "base_avg_pred": float(base_pred),
            "after_add_recent_content_review_gap2d": float(cf_pred),
            "delta": float(cf_pred - base_pred),
        }
    }
    with open(paths.metrics_dir / "insights_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved figures:")
    print("-", paths.figures_dir / "fatigue_cohort.png")
    print("-", paths.figures_dir / "time_gap_sensitivity.png")
    print("Saved summary:", paths.metrics_dir / "insights_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
