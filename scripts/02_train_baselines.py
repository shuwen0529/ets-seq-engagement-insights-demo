from __future__ import annotations
import json
import pandas as pd

from src.config import Paths, load_cfg, ensure_dirs
from src.sequence_builder import make_static_features
from src.baselines import train_logreg_baseline, train_xgb_optional
from src.eval import evaluate_binary


def main():
    paths = Paths()
    cfg = load_cfg(paths.config_path)
    ensure_dirs(paths)

    users = pd.read_csv(paths.data_dir / "synthetic_users.csv")
    events = pd.read_csv(paths.data_dir / "synthetic_events.csv")

    split = json.load(open(paths.metrics_dir / "split_users.json", "r", encoding="utf-8"))
    train_users = set(split["train_users"])
    val_users = set(split["val_users"])

    static_df = make_static_features(events, users)
    train_df = static_df[static_df["user_id"].isin(train_users)].copy()
    val_df = static_df[static_df["user_id"].isin(val_users)].copy()

    y_val = val_df["outcome_label"].astype(int).values
    q = cfg["outputs"]["top_quantile"]

    # LogReg
    logreg_model = train_logreg_baseline(train_df, seed=cfg["seed"])
    Xv_lr = val_df.drop(columns=["outcome_label", "user_id"])
    p_lr = logreg_model.predict_proba(Xv_lr)[:, 1]
    logreg_metrics = evaluate_binary(y_val, p_lr, q=q)

    # XGBoost (optional)
    xgb_model = train_xgb_optional(train_df, seed=cfg["seed"])
    xgb_metrics = None
    if xgb_model is not None:
        Xt = train_df.drop(columns=["outcome_label", "user_id"])
        Xv = val_df.drop(columns=["outcome_label", "user_id"])
        Xt2 = pd.get_dummies(Xt, columns=["segment", "platform_pref"], drop_first=True)
        Xv2 = pd.get_dummies(Xv, columns=["segment", "platform_pref"], drop_first=True)
        Xv2 = Xv2.reindex(columns=xgb_model._train_columns, fill_value=0)
        p_xgb = xgb_model.predict_proba(Xv2)[:, 1]
        xgb_metrics = evaluate_binary(y_val, p_xgb, q=q)

    out = {"logreg": logreg_metrics, "xgb": xgb_metrics}
    with open(paths.metrics_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Baseline metrics (same split, same metrics):")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
