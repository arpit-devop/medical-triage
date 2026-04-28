"""
Bidirectional LSTM with Attention classifier for medical specialty prediction.

Architecture (matches the trained state_dict in trained/best_lstm.pt.zip):
  Input (integer-encoded token sequence)
    → Embedding(17895, 128, padding_idx=0)
    → Dropout(0.3)
    → BiLSTM(128→256, 2 layers, bidirectional)
    → Attention Layer (linear 512→1, softmax over timesteps)
    → LayerNorm(512)
    → FC(512 → 22)

Note: This architecture matches the EXACT state dict keys saved by the
Kaggle training notebook.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention for medical text classification.

    Parameters
    ----------
    vocab_size  : int   – size of the vocabulary (incl. PAD=0 and UNK=1)
    embed_dim   : int   – dimensionality of word embeddings (default 128)
    hidden_dim  : int   – LSTM hidden dimension per direction (default 256)
    num_layers  : int   – stacked LSTM layers (default 2)
    num_classes : int   – number of output classes (default 22)
    dropout     : float – dropout probability (default 0.3)
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=22, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout    = nn.Dropout(dropout)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Attention: linear over bidirectional hidden (hidden_dim * 2)
        self.attn_w     = nn.Linear(hidden_dim * 2, 1)
        self.norm       = nn.LayerNorm(hidden_dim * 2)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded     = self.dropout(self.embedding(x))
        lstm_out, _  = self.lstm(embedded)       # (B, T, 2*hidden_dim)

        # Attention
        scores  = self.attn_w(lstm_out).squeeze(-1)           # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)             # (B, 2*hidden_dim)

        context = self.norm(context)
        return self.fc(context)


if __name__ == "__main__":
    # Quick test
    model = BiLSTMClassifier(
        vocab_size=17895, embed_dim=128, hidden_dim=256,
        num_layers=2, num_classes=22
    )
    print("BiLSTM with Attention Model:")
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"\nState dict keys:")
    for k, v in model.state_dict().items():
        print(f"  {k}: {v.shape}")
