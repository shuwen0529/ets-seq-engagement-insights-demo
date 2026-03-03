from __future__ import annotations
import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.config import Paths, load_cfg, ensure_dirs
from src.utils import set_seed
from src.data_gen import generate_users, generate_events

def main():
    paths = Paths()
    cfg = load_cfg(paths.config_path)
    ensure_dirs(paths)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    users = generate_users(cfg["data"]["n_users"], rng)
    users, events = generate_events(
        users,
        cfg["data"]["min_events"],
        cfg["data"]["max_events"],
        cfg["data"]["start_date"],
        cfg["data"]["n_days"],
        rng
    )

    users.to_csv(paths.data_dir / "synthetic_users.csv", index=False)
    events.to_csv(paths.data_dir / "synthetic_events.csv", index=False)

    print("Saved:")
    print("-", paths.data_dir / "synthetic_users.csv")
    print("-", paths.data_dir / "synthetic_events.csv")

    # Fixed train/val split by user (stratified by outcome)
    train_users, val_users = train_test_split(
        users["user_id"].values,
        test_size=0.2,
        random_state=cfg["seed"],
        stratify=users["outcome_label"].values
    )
    split = {"train_users": train_users.tolist(), "val_users": val_users.tolist()}
    split_path = paths.metrics_dir / "split_users.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print("Saved split:", split_path)

if __name__ == "__main__":
    main()
