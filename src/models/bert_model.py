"""
Fine-tunable BERT/BioBERT classifier for medical specialty prediction.

Architecture (matches the trained state_dict in trained/best_biobert.pt.zip):
  Input IDs + Attention Mask
    → BioBERT Encoder (12 transformer layers, 768 hidden)
    → [CLS] token embedding (B, 768)
    → FC(768 → 22)

Note: This simpler architecture matches the EXACT state dict keys saved
by the Kaggle training notebook. The notebook's Cell 16 shows a later
version with dropout/norm/two FC layers, but the actual saved model
uses a single FC layer.

Default model_name: "dmis-lab/biobert-v1.1" (biomedical pre-training).
Falls back to "bert-base-uncased" if BioBERT is unavailable.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTMedicalClassifier(nn.Module):
    """
    Fine-tuned BERT/BioBERT for medical specialty classification.

    Parameters
    ----------
    model_name     : str   – HuggingFace model identifier
    num_classes    : int   – number of output classes (default 22)
    dropout        : float – dropout probability (default 0.3)
    """

    def __init__(self, model_name="dmis-lab/biobert-v1.1", num_classes=22,
                 dropout=0.3):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size   # 768
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]     # [CLS] token  (B, 768)
        return self.fc(cls_out)


if __name__ == "__main__":
    # Quick test
    model = BERTMedicalClassifier(num_classes=22)
    print("BERTMedicalClassifier:")
    print(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"\nNon-BERT state dict keys:")
    for k, v in model.state_dict().items():
        if not k.startswith('bert.'):
            print(f"  {k}: {v.shape}")
