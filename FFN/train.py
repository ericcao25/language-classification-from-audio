import argparse
import copy
import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import preprocess


class PreprocessedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["x"], item["y"]  # x: [T,80], y: int

class FNNBaseline(nn.Module):
    def __init__(self, num_classes, input_dim=80, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class FNN_BN(nn.Module):
    def __init__(self, num_classes, input_dim=80, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def shards_exist(region_root):
    for split in ("train", "validation", "test"):
        pattern = os.path.join(region_root, f"{split}_shards", f"{split}_shard_*.pt")
        if not glob.glob(pattern):
            return False
    return True

def num_classes_for_region(region, json_path="fleurs_regions.json"):
    with open(json_path, "r") as f:
        all_regions = json.load(f)
    return len(all_regions[region]["configs"])

def load_sharded_dataset(shard_dir, prefix):
    files = sorted(glob.glob(os.path.join(shard_dir, f"{prefix}_shard_*.pt")))
    all_data = []
    for f in files:
        all_data.extend(torch.load(f, weights_only=False))
    return all_data

def collate_fnn(batch):
    xs, ys = zip(*batch)
    pooled = [x.mean(dim=0) for x in xs]         # [80]
    X = torch.stack(pooled, dim=0)               # [B,80]
    y = torch.tensor(ys, dtype=torch.long)       # [B]
    return X, y

def build_loaders(save_root, batch_size):
    train_shards = os.path.join(save_root, "train_shards")
    val_shards = os.path.join(save_root, "validation_shards")
    test_shards = os.path.join(save_root, "test_shards")
    train_data = load_sharded_dataset(train_shards, "train")
    val_data = load_sharded_dataset(val_shards, "validation")
    test_data = load_sharded_dataset(test_shards, "test")
    train_loader = DataLoader(PreprocessedDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fnn)
    val_loader = DataLoader(PreprocessedDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fnn)
    test_loader = DataLoader(PreprocessedDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fnn)
 
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        if return_preds:
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    if return_preds:
        return total_loss/total, correct/total, torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    return total_loss/total, correct/total

def run_training(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, tag, track_best=False):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    for epoch in range(1, num_epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"[{tag}] Epoch {epoch:02d}/{num_epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if track_best and va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"  best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    if track_best and best_state:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint: epoch={best_epoch}, val_acc={best_val_acc:.4f}")
    return history, best_epoch

def plot_curves(history, title_prefix, best_epoch=None):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    if best_epoch and best_epoch > 0:
        plt.scatter([best_epoch], [history["val_loss"][best_epoch - 1]], color="red", zorder=5, label=f"best @ {best_epoch}")
        plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"{title_prefix}: Loss per epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    if best_epoch and best_epoch > 0:
        plt.scatter([best_epoch], [history["val_acc"][best_epoch - 1]], color="red", zorder=5, label=f"best @ {best_epoch}")
        plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(f"{title_prefix}: Accuracy per epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main(region, batch_size=256, save_root="fleurs_preprocessed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    region_root = os.path.join(save_root, region)
    if shards_exist(region_root):
        print(f"Shards already exist at {region_root}, skipping preprocessing.")
    else:
        preprocess(region, save_root=save_root)
    
    num_classes = num_classes_for_region(region)
    train_loader, val_loader, test_loader = build_loaders(region_root, batch_size)
    X, y = next(iter(train_loader))
    
    # Baseline feed-forward network
    model_base = FNNBaseline(num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_base.parameters(), lr=1e-3)
    history_base, best_epoch_base = run_training(model_base, train_loader, val_loader, optimizer, criterion, device, 10, "BASE")
    te_loss, te_acc = evaluate(model_base, test_loader, criterion, device)
    print(f"[BASE] TEST | loss={te_loss:.4f} acc={te_acc:.4f}")
    plot_curves(history_base, "BASELINE FFN", best_epoch_base)

    # Upgraded FNN: BatchNorm, AdamW, 30 epochs
    model_up = FNN_BN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_up.parameters(), lr=3e-4, weight_decay=1e-2)
    history_up, best_epoch_up = run_training(model_up, train_loader, val_loader, optimizer, criterion, device, 30, "UP", track_best=True)
    te_loss, te_acc = evaluate(model_up, test_loader, criterion, device)
    print(f"[UPGRADED FFN] TEST | loss={te_loss:.4f} acc={te_acc:.4f}")
    plot_curves(history_up, "UPGRADED FFN", best_epoch_up)

    # Upgraded FNN: longer/more regularized training
    model_up2 = FNN_BN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_up2.parameters(), lr=1e-4, weight_decay=2e-2)
    history_up2, best_epoch_up2 = run_training(model_up2, train_loader, val_loader, optimizer, criterion, device, 60, "UP2", track_best=True)
    te_loss, te_acc, y_true, y_pred = evaluate(model_up2, test_loader, criterion, device, return_preds=True)
    print(f"[UPGRADED FFN 2] TEST | loss={te_loss:.4f} acc={te_acc:.4f} (sklearn acc={accuracy_score(y_true, y_pred):.4f})")
    plot_curves(history_up2, "UPGRADED FFN 2", best_epoch_up2)
    plot_confusion_matrix(y_true, y_pred, "UPGRADED FFN 2 Test Confusion Matrix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLEURS regional language-ID pipeline")
    parser.add_argument("--region", choices=["eastern_europe", "western_europe", "central_asia_middle_east_north_africa", "sub_saharan_africa", "south_asia", "south_east_asia", "cjk"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_root", type=str, default="fleurs_preprocessed")
    args = parser.parse_args()
 
    main(args.region, batch_size=args.batch_size, save_root=args.save_root)