from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class SeqDataset(Dataset):
    def __init__(self, X_events, X_gaps, mask, y):
        self.Xe = torch.tensor(X_events, dtype=torch.long)
        self.Xg = torch.tensor(X_gaps, dtype=torch.float32)
        self.m = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i):
        return self.Xe[i], self.Xg[i], self.m[i], self.y[i]

def make_loaders_from_indices(Xe, Xg, m, y, train_idx, val_idx, batch_size):
    train_ds = SeqDataset(Xe[train_idx], Xg[train_idx], m[train_idx], y[train_idx])
    val_ds = SeqDataset(Xe[val_idx], Xg[val_idx], m[val_idx], y[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl

def train_model(model, train_dl, val_dl, lr, epochs, patience, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for Xe, Xg, m, y in tqdm(train_dl, desc=f"epoch {ep}", leave=False):
            Xe, Xg, m, y = Xe.to(device), Xg.to(device), m.to(device), y.to(device)
            _p, logits = model(Xe, Xg, m)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for Xe, Xg, m, y in val_dl:
                Xe, Xg, m = Xe.to(device), Xg.to(device), m.to(device)
                p, _ = model(Xe, Xg, m)
                ys.append(y.numpy())
                ps.append(p.detach().cpu().numpy())

        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        auc = roc_auc_score(yv, pv)

        if auc > best_auc + 1e-4:
            best_auc = float(auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc
