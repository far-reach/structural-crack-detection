import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("results/metrics/history.json") as f:
    history = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train", linewidth=2)
ax1.plot(history["val_loss"],   label="Val",   linewidth=2)
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mark the spike region
spike_start = 12
spike_end   = 16
ax1.axvspan(spike_start, spike_end, alpha=0.1, color='red',
            label='LR decay effect')
ax1.annotate('LR decay\n(epoch 7)',
             xy=(7, max(history["val_loss"][:10])),
             xytext=(9, max(history["val_loss"])*0.85),
             fontsize=8, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1))

ax2.plot(history["train_acc"], label="Train", linewidth=2)
ax2.plot(history["val_acc"],   label="Val",   linewidth=2)
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Structural Damage Detection — Training Results", fontweight="bold")
plt.tight_layout()
plt.savefig("results/plots/training_curves_multiclass.png", dpi=150)
plt.savefig("results/plots/training_curves_multiclass.jpg", dpi=150)
print("Saved training curves with epoch labels and spike annotation.")