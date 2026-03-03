from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

def make_vocab(events: pd.Series) -> Dict[str, int]:
    uniq = sorted(events.unique().tolist())
    vocab = {"<PAD>": 0}
    for i, e in enumerate(uniq, start=1):
        vocab[e] = i
    return vocab

def build_user_sequences(
    events_df: pd.DataFrame,
    users_df: pd.DataFrame,
    vocab: Dict[str, int],
    max_len: int,
    min_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    events_df = events_df.sort_values(["user_id", "event_timestamp"]).copy()
    seq_rows, gap_rows, mask_rows, y_rows = [], [], [], []
    kept_users: List[str] = []

    for user_id, g in events_df.groupby("user_id"):
        ev = g["event_type"].map(vocab).to_numpy(dtype=np.int64)
        ts = pd.to_datetime(g["event_timestamp"]).to_numpy()

        if len(ev) < min_len:
            continue

        gaps = np.zeros(len(ev), dtype=np.float32)
        if len(ev) > 1:
            dt = (ts[1:] - ts[:-1]) / np.timedelta64(1, "D")
            gaps[1:] = np.clip(dt.astype(np.float32), 0, 60)

        if len(ev) > max_len:
            ev = ev[-max_len:]
            gaps = gaps[-max_len:]

        L = len(ev)
        pad = max_len - L

        x = np.pad(ev, (pad, 0), constant_values=0)
        gpad = np.pad(gaps, (pad, 0), constant_values=0.0)
        m = np.pad(np.ones(L, dtype=np.float32), (pad, 0), constant_values=0.0)

        label = int(users_df.loc[users_df["user_id"] == user_id, "outcome_label"].iloc[0])

        seq_rows.append(x)
        gap_rows.append(gpad)
        mask_rows.append(m)
        y_rows.append(label)
        kept_users.append(user_id)

    return (
        np.stack(seq_rows),
        np.stack(gap_rows),
        np.stack(mask_rows),
        np.array(y_rows, dtype=np.int64),
        kept_users
    )

def make_static_features(events_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.sort_values(["user_id", "event_timestamp"]).copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    agg = df.groupby("user_id").agg(
        total_events=("event_type", "size"),
        n_assess=("event_type", lambda x: (x == "assessment_attempt").sum()),
        n_practice=("event_type", lambda x: (x == "practice_session").sum()),
        last_ts=("event_timestamp", "max"),
    ).reset_index()
    as_of = df["event_timestamp"].max()
    agg["days_since_last"] = (as_of - agg["last_ts"]).dt.days.astype(int)

    out = users_df[["user_id", "segment", "platform_pref", "baseline_motivation", "outcome_label"]].merge(
        agg.drop(columns=["last_ts"]), on="user_id", how="left"
    )
    return out
