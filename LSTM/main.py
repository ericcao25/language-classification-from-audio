import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_fleurs_dataset, N_MELS
from lstm import LSTMLanguageClassifier, train_step, val_step, evaluate_per_class, _get_state_dict


def _passthrough_collate(x):
    """Module-level passthrough collate — picklable unlike a lambda."""
    return x

def main(region, num_epochs=100, batch_size=64, debug=False):
    EARLY_STOP_PAT = 20    # stop if val accuracy doesn't improve for this many epochs
    CKPT_INTERVAL = 20    # save a periodic checkpoint every N epochs
    NUM_GPUS = max(torch.cuda.device_count(), 1)
    BATCH_SIZE = batch_size * NUM_GPUS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"GPUs: {NUM_GPUS}  |  Effective batch size: {BATCH_SIZE}  |  Device: {device}")

    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, id2config = load_fleurs_dataset(region, batch_size=BATCH_SIZE)
    print(f"Batches per epoch — train: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}")

    num_classes = len(id2config)
    model = LSTMLanguageClassifier(
        input_dim=N_MELS,
        hidden_dim=64,
        num_layers=1,
        num_classes=num_classes,
        bidirectional=True,
        dropout=0.6,
    )
    if NUM_GPUS > 1:
        print(f"Using DataParallel across {NUM_GPUS} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer    = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    best_val_acc      = 0.0
    epochs_no_improve = 0
    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    os.makedirs(region, exist_ok=True)
    print()
    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer, device)
        val_loss,   val_acc   = val_step(model, val_loader, device)
        lr_scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        # Best-model checkpoint
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(_get_state_dict(model), f"{region}/lstm_best.pth")
            print(f"  -> New best val accuracy: {best_val_acc:.4f}. Checkpoint saved.")
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if epoch % CKPT_INTERVAL == 0:
            torch.save(_get_state_dict(model), f"{region}/lstm_epoch{epoch:03d}.pth")

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PAT:
            print(f"\nNo improvement for {EARLY_STOP_PAT} epochs. Stopping early at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe for cluster environments
    import matplotlib.pyplot as plt

    epochs_range = range(len(train_losses))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, label="Train Accuracy")
    ax2.plot(epochs_range, val_accs,   label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{region}/training_curves.png", dpi=150)
    print(f"Training curves saved to {region}/training_curves.png")

    # Final evaluation on test set using the best checkpoint
    print("\nLoading best checkpoint for test evaluation...")
    underlying = model.module if isinstance(model, nn.DataParallel) else model
    underlying.load_state_dict(torch.load(f"{region}/lstm_best.pth", map_location=device))

    per_class = evaluate_per_class(model, test_loader, device, id2config)
    overall_correct = sum(c for c, _, __ in per_class.values())
    overall_total   = sum(t for _, t, __ in per_class.values())

    overall_acc = overall_correct / overall_total
    print(f"\nTest accuracy: {overall_acc:.4f}")
    print("\nPer-class breakdown:")
    for cfg, (correct, total, acc) in per_class.items():
        bar = "█" * int(acc * 20)
        print(f"  {cfg:20s}  {acc:.4f}  {bar:20s}  ({correct}/{total})")

    # Save per-class accuracy as a bar chart
    cfgs  = list(per_class.keys())
    accs  = [per_class[c][2] for c in cfgs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cfgs, accs, color="#2196F3", edgecolor="white")

    # Label each bar with its accuracy value
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(overall_acc, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Overall accuracy ({overall_acc:.3f})")
    ax.axhline(1 / num_classes, color="gray", linestyle=":", linewidth=1.2,
               label=f"Chance ({1/num_classes:.3f})")
    ax.set_ylim(0, min(1.0, max(accs) + 0.1))
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class Test Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{region}/per_class_accuracy.png", dpi=150)
    print(f"Per-class accuracy chart saved to {region}/per_class_accuracy.png")

    # Save final model
    torch.save(_get_state_dict(model), f"{region}/lstm_model.pth")
    print(f"\nFinal model saved to {region}/lstm_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLEURS CNN language classification")
    parser.add_argument("--region", type=str, choices=["eastern_europe", "western_europe", "central_asia_middle_east_north_africa", "sub_saharan_africa", "south_asia", "south_east_asia", "cjk"])
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args.region, num_epochs=args.num_epochs, batch_size=args.batch_size)
