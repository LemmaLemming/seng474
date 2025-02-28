import os
import numpy as np
from sklearn.svm import SVC
from utils.a1_k_fold_cv import k_fold_cross_validation  # Import custom k-fold function

# Load preprocessed data
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load C values
C_values = np.load("utils/c_values.npy")

# k-Fold Cross-Validation
k = 5
C_scores = {}

print("Performing k-Fold Cross-Validation...")

for C in C_values:
    model = SVC(kernel="linear", C=C)
    mean_score, scores = k_fold_cross_validation(model, X_train, y_train, k=k)
    C_scores[C] = mean_score
    print(f"C={C:.6f}, Mean Accuracy={mean_score:.4f}")

# Save best C for later training
best_C = max(C_scores, key=C_scores.get)
np.save("utils/best_C.npy", best_C)
print(f"\nBest C value: {best_C}")
print(f"Best C value saved to 'utils/best_C.npy'")

# Save C values and accuracies to a txt file
txt_file = "utils/c_values_accuracy.txt"
with open(txt_file, "w") as f:
    f.write("C Values and Mean Accuracy Scores\n")
    f.write("-" * 40 + "\n")
    for C, score in C_scores.items():
        f.write(f"C={C:.6f}, Mean Accuracy={score:.4f}\n")

print(f"C values and accuracies saved to '{txt_file}'")
