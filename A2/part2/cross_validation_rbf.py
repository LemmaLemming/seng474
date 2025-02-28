import numpy as np
from sklearn.svm import SVC
from utils.a1_k_fold_cv import k_fold_cross_validation  # Import k-fold CV function
import json

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# hyperparameter ranges
gamma_values = np.logspace(-5, 2, num=10)  # 10 values from 10^-5 to 10^2
C_values = np.logspace(-3, 3, num=10)      # 10 values from 10^-3 to 10^3

# Save gamma values and C values
np.save("gamma_values.npy", gamma_values)
np.save("C_values.npy", C_values)

# Dictionary to store the best C for each y
best_C_per_gamma = {}
cv_scores_per_gamma = {}

print("Performing double cross-validation for RBF SVM...")

for gamma in gamma_values:
    C_scores = {}
    
    for C in C_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        mean_score, _ = k_fold_cross_validation(model, X_train, y_train, k=5)
        C_scores[C] = mean_score
        print(f"y={gamma:.10f}, C={C:.6f}, CV Accuracy={mean_score:.4f}")


    # Select best y
    best_C = max(C_scores, key=C_scores.get)
    best_C_per_gamma[gamma] = best_C
    cv_scores_per_gamma[gamma] = C_scores[best_C]

# Select the best y based on CV performance
best_gamma = max(cv_scores_per_gamma, key=cv_scores_per_gamma.get)
best_C_for_best_gamma = best_C_per_gamma[best_gamma]

print(f"\nBest y: {best_gamma}, Best C: {best_C_for_best_gamma}")

# Save all best (y, Cy) pairs
with open("best_C_per_gamma.json", "w") as f:
    json.dump(best_C_per_gamma, f)

# Save best y and corresponding Cy
np.save("best_gamma.npy", best_gamma)
np.save("best_C_for_best_gamma.npy", best_C_for_best_gamma)

print("Saved all best (y, Cy) pairs and best overall y, C for later training.")
