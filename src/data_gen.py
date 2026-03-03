from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

EVENT_TYPES = [
    "practice_session",
    "assessment_attempt",
    "score_feedback",
    "content_review",
    "re_engage",
]

def _pick_segment(rng: np.random.Generator) -> str:
    return rng.choice(["high", "mid", "low"], p=[0.25, 0.50, 0.25])

def _baseline_motivation(segment: str, rng: np.random.Generator) -> float:
    if segment == "high":
        return float(rng.normal(1.2, 0.2))
    if segment == "mid":
        return float(rng.normal(0.6, 0.2))
    return float(rng.normal(0.1, 0.25))

def generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(n_users):
        seg = _pick_segment(rng)
        pref = rng.choice(["web", "mobile"], p=[0.55, 0.45])
        mot = _baseline_motivation(seg, rng)
        rows.append((f"U{i:05d}", seg, pref, mot))
    df = pd.DataFrame(rows, columns=["user_id", "segment", "platform_pref", "baseline_motivation"])
    df["outcome_label"] = 0
    return df

def _next_event_type(prev: str | None, rng: np.random.Generator) -> str:
    if prev is None:
        return rng.choice(["practice_session", "content_review"], p=[0.7, 0.3])
    if prev == "practice_session":
        return rng.choice(["practice_session", "assessment_attempt", "content_review"], p=[0.35, 0.45, 0.20])
    if prev == "assessment_attempt":
        return rng.choice(["score_feedback", "content_review", "practice_session"], p=[0.55, 0.25, 0.20])
    if prev == "score_feedback":
        return rng.choice(["content_review", "practice_session", "re_engage"], p=[0.45, 0.40, 0.15])
    if prev == "content_review":
        return rng.choice(["practice_session", "assessment_attempt", "re_engage"], p=[0.40, 0.45, 0.15])
    return rng.choice(EVENT_TYPES)

def _sample_gap_days(rng: np.random.Generator, segment: str) -> int:
    if segment == "high":
        return int(max(1, rng.poisson(2)))
    if segment == "mid":
        return int(max(1, rng.poisson(4)))
    return int(max(1, rng.poisson(7)))

def generate_events(
    users_df: pd.DataFrame,
    min_events: int,
    max_events: int,
    start_date: str,
    n_days: int,
    rng: np.random.Generator
):
    start_dt = datetime.fromisoformat(start_date)
    all_rows = []

    for _, u in users_df.iterrows():
        n_ev = int(rng.integers(min_events, max_events + 1))
        t = start_dt + timedelta(days=int(rng.integers(0, max(1, n_days // 3))))
        prev = None

        gaps: List[int] = []
        timestamps: List[datetime] = []

        for _j in range(n_ev):
            ev = _next_event_type(prev, rng)

            plat = u["platform_pref"]
            if rng.random() < 0.25:
                plat = "mobile" if plat == "web" else "web"

            score = np.nan
            if ev == "assessment_attempt":
                base = 65 if u["segment"] == "low" else 72 if u["segment"] == "mid" else 78
                score = float(np.clip(rng.normal(base, 10), 0, 100))

            all_rows.append((u["user_id"], t, ev, plat, score, u["segment"]))
            timestamps.append(t)

            gap = _sample_gap_days(rng, u["segment"])
            gaps.append(gap)
            t = t + timedelta(days=gap)
            prev = ev

        # --- Synthetic "truth" label (amplified sequence effects) ---
        mot = float(u["baseline_motivation"])

        early_gaps = gaps[:3] if len(gaps) >= 3 else gaps
        early_avg = float(np.mean(early_gaps)) if early_gaps else 5.0
        momentum_boost = 0.6 * (1.0 / (1.0 + early_avg))  # shorter early gaps => higher boost

        end_time = timestamps[-1]
        in_last = sum(1 for ts in timestamps if (end_time - ts).days <= 14)
        fatigue_penalty = 0.12 * max(0, in_last - 3)

        gap_avg = float(np.mean(gaps))
        decay_penalty = 0.02 * max(0, gap_avg - 4)

        # Reduce static dominance; amplify temporal effects
        seg_base = {"high": 0.25, "mid": 0.05, "low": -0.10}[u["segment"]]
        logit = (
            seg_base
            + 0.45 * mot
            + 2.2 * momentum_boost
            - 1.35 * fatigue_penalty
            - 1.25 * decay_penalty
            + float(rng.normal(0, 0.15))
        )
        p = 1 / (1 + np.exp(-logit))
        label = int(rng.random() < p)
        users_df.loc[users_df["user_id"] == u["user_id"], "outcome_label"] = label

    events_df = pd.DataFrame(
        all_rows,
        columns=["user_id", "event_timestamp", "event_type", "platform", "score", "segment"]
    )
    return users_df, events_df
