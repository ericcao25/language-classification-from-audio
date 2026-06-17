import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMLanguageClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, T, F] padded log-mel features
        lengths : [B] int64 — true sequence lengths
        returns   logits [B, num_classes]
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        # h_n: [num_layers * num_directions, B, hidden_dim]
        if self.lstm.bidirectional:
            feats = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        else:
            feats = h_n[-1]  # [B, H]

        return self.classifier(feats)


def _get_state_dict(model):
    """Unwrap DataParallel before saving so checkpoints are portable."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def train_step(model, dataloader, optimizer, device, class_weights=None):
    """
    One full training epoch. Returns (avg_loss, accuracy).
    class_weights: optional [num_classes] tensor on the correct device,
                   passed to cross_entropy to upweight underrepresented classes.
    """
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)
        loss = nn.functional.cross_entropy(
            logits, y,
            weight=class_weights,
            label_smoothing=0.1,
        ).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs             = y.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def val_step(model, dataloader, device, class_weights=None):
    """
    One full validation pass. Returns (avg_loss, accuracy).
    class_weights: optional [num_classes] tensor, same as train_step.
    """
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)
        loss = nn.functional.cross_entropy(
            logits, y,
            weight=class_weights,
            label_smoothing=0.1,
        ).mean()

        bs             = y.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_per_class(model, dataloader, device, id2config):
    """
    Returns a dict mapping config string -> (correct, total, accuracy)
    so you can see which languages the model struggles with.
    """
    model.eval()
    correct_per_class = {}
    total_per_class   = {}

    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        preds = model(x, lengths).argmax(dim=-1)

        for true_idx, pred_idx in zip(y.tolist(), preds.tolist()):
            cfg = id2config[true_idx]
            total_per_class[cfg]   = total_per_class.get(cfg, 0) + 1
            correct_per_class[cfg] = correct_per_class.get(cfg, 0) + int(true_idx == pred_idx)

    return {
        cfg: (correct_per_class[cfg], total_per_class[cfg],
              correct_per_class[cfg] / total_per_class[cfg])
        for cfg in sorted(total_per_class)
    }
