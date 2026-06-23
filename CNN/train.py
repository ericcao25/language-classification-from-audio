import argparse
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from collections import Counter
from datasets import Dataset, concatenate_datasets, load_dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_configs import FLEURS_GROUP_INFO

TARGET_SR = 16000
N_MELS = 80


class CNN(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, num_classes=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.conv(x)
        x = x.mean(dim=-1)     # global average pooling -> [B, hidden*2]
        return self.classifier(x)


def load_region_configs(region):
    return FLEURS_GROUP_INFO[region]

def load_data(langs, examples_per_lang=1300):
    """Stream each language's training split and partition into train/val/test
    by example index: first 100 -> test, next 100 -> val, rest -> train."""
    train_list, val_list, test_list = [], [], []

    for i, lang in enumerate(langs):
        tr = load_dataset("google/fleurs", lang, split="train",
                          streaming=True, trust_remote_code=True)

        training, validation, testing = [], [], []
        for j, ex in enumerate(tr):
            if j >= examples_per_lang:
                break
            ex["lang_id"] = i
            if j < 100:
                testing.append(ex)
            elif j < 200:
                validation.append(ex)
            else:
                training.append(ex)
            if j % 100 == 0:
                print(f"{lang}: {j} examples downloaded")

        train_list.append(Dataset.from_list(training))
        val_list.append(Dataset.from_list(validation))
        test_list.append(Dataset.from_list(testing))
        del training, validation, testing # prevent OOM

    train_ds = concatenate_datasets(train_list).shuffle(seed=42)
    val_ds = concatenate_datasets(val_list).shuffle(seed=42)
    test_ds = concatenate_datasets(test_list).shuffle(seed=42)
    print("Train size:", len(train_ds))
    print("Validation size:", len(val_ds))
    print("Test size:", len(test_ds))

    return train_ds, val_ds, test_ds

def make_transforms():
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, hop_length=160, n_mels=N_MELS,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    return mel, amp_to_db

def waveform_to_logmel(waveform, mel, amp_to_db):
    """waveform: [1, N] -> [T, F] log-mel spectrogram."""
    m = mel(waveform)
    m = amp_to_db(m)
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.squeeze(0).transpose(0, 1).contiguous()

def make_collate_fn(labels, mel, amp_to_db):
    def collate_fn(batch):
        feats_list, lengths, target_labels = [], [], []
        for ex in batch:
            wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
            feats = waveform_to_logmel(wav, mel, amp_to_db)
            feats = (feats - feats.mean(dim=0, keepdim=True)) / (feats.std(dim=0, keepdim=True) + 1e-8)
            feats_list.append(feats)
            lengths.append(feats.shape[0])
            target_labels.append(labels.index(ex["lang_id"]))
        padded = pad_sequence(feats_list, batch_first=True)   # [B, Tmax, F]
        return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(target_labels, dtype=torch.long)
    return collate_fn

def build_loaders(train_ds, val_ds, test_ds, collate_fn, batch_size=64):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        correct += (logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total, correct / total

def run_training(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    return history

def plot_curves(history, region):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    tick_step = max(1, len(epochs) // 10)
    title = region.replace("_", " ").title()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(f"Language Classification CNN for {title}: Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(epochs) + 1, tick_step))
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.legend()
    plt.title(f"Language Classification CNN for {title}: Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(epochs) + 1, tick_step))
    plt.show()

def plot_prediction_distribution(model, test_loader, device, lang_names):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x, lengths, y in test_loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())

    counts = Counter(all_preds)
    values = [counts.get(i, 0) for i in range(len(lang_names))]

    plt.figure()
    plt.bar(lang_names, values)
    plt.xlabel("Predicted Language")
    plt.ylabel("Count")
    plt.title("Distribution of Language Predictions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def main(region, num_epochs=5, batch_size=64, regions_json="../fleurs_regions.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region_info = load_region_configs(region)
    langs = region_info["configs"]
    lang_names = region_info["names"]
    train_ds, val_ds, test_ds = load_data(langs)

    labels = train_ds.unique("lang_id")
    mel, amp_to_db = make_transforms()
    collate_fn = make_collate_fn(labels, mel, amp_to_db)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, collate_fn, batch_size)

    model = CNN(num_classes=len(langs)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    history = run_training(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
    plot_curves(history, region)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f} acc={test_acc:.4f}")

    plot_prediction_distribution(model, test_loader, device, lang_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLEURS CNN language classification")
    parser.add_argument("--region", type=str, choices=["eastern_europe", "western_europe", "central_asia_middle_east_north_africa", "sub_saharan_africa", "south_asia", "south_east_asia", "cjk"])
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args.region, num_epochs=args.num_epochs, batch_size=args.batch_size, regions_json=args.regions_json)