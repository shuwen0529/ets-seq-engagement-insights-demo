from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def _predict_batch(model, Xe, Xg, m, device):
    model.eval()
    with torch.no_grad():
        Xe_t = torch.tensor(Xe, dtype=torch.long, device=device)
        Xg_t = torch.tensor(Xg, dtype=torch.float32, device=device)
        m_t  = torch.tensor(m,  dtype=torch.float32, device=device)
        p, _ = model(Xe_t, Xg_t, m_t)
        return p.detach().cpu().numpy()

def cohort_fatigue(events_df: pd.DataFrame, users_df: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    df = events_df.sort_values(["user_id", "event_timestamp"]).copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    end = df.groupby("user_id")["event_timestamp"].max().rename("end_ts")
    df = df.merge(end, on="user_id", how="left")
    df["in_window"] = (df["end_ts"] - df["event_timestamp"]).dt.days <= window_days

    w = df[df["in_window"]].groupby("user_id").size().rename("events_last_window").reset_index()
    out = users_df[["user_id", "outcome_label", "segment"]].merge(w, on="user_id", how="left").fillna({"events_last_window": 0})
    out["events_last_window"] = out["events_last_window"].astype(int)

    summary = out.groupby("events_last_window").agg(
        n_users=("user_id", "size"),
        actual_rate=("outcome_label", "mean")
    ).reset_index()
    return summary

def counterfactual_add_recent_event(model, Xe, Xg, m, add_event_id: int, add_gap_days: float, device):
    Xe2, Xg2, m2 = Xe.copy(), Xg.copy(), m.copy()
    for i in range(len(Xe2)):
        Xe2[i, :-1] = Xe2[i, 1:]
        Xg2[i, :-1] = Xg2[i, 1:]
        m2[i, :-1]  = m2[i, 1:]
        Xe2[i, -1]  = add_event_id
        Xg2[i, -1]  = float(add_gap_days)
        m2[i, -1]   = 1.0
    return _predict_batch(model, Xe2, Xg2, m2, device)

def time_gap_sensitivity_sweep(model, Xe, Xg, m, device, gap_values=(1,3,7,14), n_samples=1200):
    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(len(Xe)), size=min(n_samples, len(Xe)), replace=False)
    Xe_s, Xg_s, m_s = Xe[idx], Xg[idx], m[idx]

    base = _predict_batch(model, Xe_s, Xg_s, m_s, device)
    rows = []
    for g in gap_values:
        Xg_mod = Xg_s.copy()
        Xg_mod[:, -1] = float(g)
        p = _predict_batch(model, Xe_s, Xg_mod, m_s, device)
        rows.append((g, float(p.mean()), float((p - base).mean())))
    return pd.DataFrame(rows, columns=["gap_days_set_at_last_step", "avg_pred_prob", "avg_delta_vs_base"])

def plot_curve(df: pd.DataFrame, xcol: str, ycol: str, title: str, out_png):
    plt.figure()
    plt.plot(df[xcol], df[ycol], marker="o")
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
