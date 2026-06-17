from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_fleurs_dataset, PER_GPU_BATCH_SIZE, id2config, NUM_CLASSES
from features import train_collate_fn, val_collate_fn, test_collate_fn, N_MELS
from lstm import LSTMLanguageClassifier, train_step, val_step, evaluate_per_class, _get_state_dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_EPOCHS      = 100
EARLY_STOP_PAT  = 20    # stop if val accuracy doesn't improve for this many epochs
CKPT_INTERVAL   = 20    # save a periodic checkpoint every N epochs

# NUM_GPUS  = max(torch.cuda.device_count(), 1)
NUM_GPUS = 1
BATCH_SIZE = PER_GPU_BATCH_SIZE * NUM_GPUS
device    = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _passthrough_collate(x):
    """Module-level passthrough collate — picklable unlike a lambda."""
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(debug: bool = False):
    print(f"GPUs: {NUM_GPUS}  |  Effective batch size: {BATCH_SIZE}  |  Device: {device}")

    # Data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = load_fleurs_dataset(
        train_collate_fn=train_collate_fn,
        val_collate_fn=val_collate_fn,
        test_collate_fn=test_collate_fn,
        batch_size=BATCH_SIZE,
    )
    print(f"Batches per epoch — train: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}")

    # Model
    model = LSTMLanguageClassifier(
        input_dim=N_MELS,
        hidden_dim=64,
        num_layers=1,
        num_classes=NUM_CLASSES,
        bidirectional=True,
        dropout=0.6,
    )
    if NUM_GPUS > 1:
        print(f"Using DataParallel across {NUM_GPUS} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer & scheduler
    optimizer    = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)

    # Training loop with early stopping
    best_val_acc      = 0.0
    epochs_no_improve = 0

    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    print()
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_step(model, train_loader, optimizer, device)
        val_loss,   val_acc   = val_step(model, val_loader, device)
        lr_scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f}"
        )

        # Best-model checkpoint
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(_get_state_dict(model), "lstm_best.pth")
            print(f"  -> New best val accuracy: {best_val_acc:.4f}. Checkpoint saved.")
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if epoch % CKPT_INTERVAL == 0:
            torch.save(_get_state_dict(model), f"lstm_epoch{epoch:03d}.pth")

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PAT:
            print(f"\nNo improvement for {EARLY_STOP_PAT} epochs. Stopping early at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

    # ---------------------------------------------------------------------------
    # Plot training curves
    # ---------------------------------------------------------------------------
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
    fig.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")

    # Final evaluation on test set using the best checkpoint
    print("\nLoading best checkpoint for test evaluation...")
    underlying = model.module if isinstance(model, nn.DataParallel) else model
    underlying.load_state_dict(torch.load("lstm_best.pth", map_location=device))

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
    ax.axhline(1 / NUM_CLASSES, color="gray", linestyle=":", linewidth=1.2,
               label=f"Chance ({1/NUM_CLASSES:.3f})")
    ax.set_ylim(0, min(1.0, max(accs) + 0.1))
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class Test Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("per_class_accuracy.png", dpi=150)
    print("Per-class accuracy chart saved to per_class_accuracy.png")

    # Save final model
    torch.save(_get_state_dict(model), "lstm_model.pth")
    print("\nFinal model saved to lstm_model.pth")


if __name__ == "__main__":
    main()
