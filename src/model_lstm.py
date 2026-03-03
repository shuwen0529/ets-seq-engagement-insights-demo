import torch
import torch.nn as nn

class EngagementLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_events, x_gaps, mask=None):
        emb = self.embedding(x_events)
        gaps = x_gaps.unsqueeze(-1)
        x = torch.cat([emb, gaps], dim=-1)

        out, _ = self.lstm(x)

        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            last = out.gather(1, idx).squeeze(1)
        else:
            last = out[:, -1, :]

        last = self.dropout(last)
        logits = self.fc(last).squeeze(-1)
        return torch.sigmoid(logits), logits
