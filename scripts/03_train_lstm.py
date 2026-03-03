from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch

from src.config import Paths, load_cfg, ensure_dirs
from src.utils import set_seed
from src.sequence_builder import make_vocab, build_user_sequences
from src.model_lstm import EngagementLSTM
from src.train_lstm import make_loaders_from_indices, train_model
from src.eval import evaluate_binary


def main():
    paths = Paths()
    cfg = load_cfg(paths.config_path)
    ensure_dirs(paths)
    set_seed(cfg["seed"])

    users = pd.read_csv(paths.data_dir / "synthetic_users.csv")
    events = pd.read_csv(paths.data_dir / "synthetic_events.csv")

    split = json.load(open(paths.metrics_dir / "split_users.json", "r", encoding="utf-8"))
    train_users = set(split["train_users"])
    val_users = set(split["val_users"])

    vocab = make_vocab(events["event_type"])
    Xe, Xg, m, y, kept_users = build_user_sequences(
        events_df=events,
        users_df=users,
        vocab=vocab,
        max_len=cfg["sequence"]["max_seq_len"],
        min_len=cfg["sequence"]["min_seq_len"],
    )

    user_to_idx = {u: i for i, u in enumerate(kept_users)}
    train_idx = [user_to_idx[u] for u in kept_users if u in train_users]
    val_idx = [user_to_idx[u] for u in kept_users if u in val_users]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EngagementLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    train_dl, val_dl = make_loaders_from_indices(
        Xe, Xg, m, y,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=cfg["model"]["batch_size"],
    )

    model, best_auc = train_model(
        model, train_dl, val_dl,
        lr=cfg["model"]["lr"],
        epochs=cfg["model"]["epochs"],
        patience=cfg["model"]["patience"],
        device=device
    )

    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for Xe_b, Xg_b, m_b, y_b in val_dl:
            p, _ = model(Xe_b.to(device), Xg_b.to(device), m_b.to(device))
            ys.append(y_b.numpy())
            ps.append(p.detach().cpu().numpy())

    yv = np.concatenate(ys)
    pv = np.concatenate(ps)
    metrics = evaluate_binary(yv, pv, q=cfg["outputs"]["top_quantile"])

    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "cfg": cfg}, paths.outputs_dir / "lstm_model.pt")
    with open(paths.metrics_dir / "lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_auc": float(best_auc), "val_metrics": metrics}, f, indent=2)

    print("LSTM metrics (same split, same metrics):")
    print(json.dumps({"best_val_auc": float(best_auc), "val_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
