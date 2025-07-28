import matplotlib.pyplot as plt

methods = [
    "Inception + LSTM [15]",
    "Late fusion [7]",
    "ResNet50 + Bi-LSTM [8]",
    "K-injection network [9]",
    "VAFO FUSION",
    "OURS"
]

accuracy = [52.09, 59.04, 61.83, 66.19, 67.36, 83.30]  # ✅ Added YOUR final accuracy

colors = ['#b0c4de', '#add8e6', '#87cefa', '#4682b4', "#8cef8c", "#542cf3"]

plt.figure(figsize=(10, 6))
bars = plt.barh(methods, accuracy, color=colors)
plt.xlabel("Accuracy (%)", fontsize=12)
plt.title("Emotion Recognition Model Comparison", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Annotate accuracy on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + 0.25, f'{width:.2f}%', fontsize=10)

plt.tight_layout()
plt.show()
