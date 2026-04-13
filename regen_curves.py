import json, matplotlib.pyplot as plt

with open("results/metrics/history.json") as f:
    history = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"],   label="Val")
ax2.set_title("Accuracy"); ax2.legend()
plt.suptitle("Structural Damage Detection - Training Results")
plt.tight_layout()
plt.savefig("results/plots/training_curves_multiclass.png", dpi=150)
plt.savefig("results/plots/training_curves_multiclass.jpg", dpi=150)
print("Training curves saved.")