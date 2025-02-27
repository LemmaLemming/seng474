import numpy as np
from sklearn.svm import SVC
from utils.a1_k_fold_cv import k_fold_cross_validation  # Import a1 k-fold CV function

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load γ values
gamma_values = np.load("utils/gamma_values.npy")

# Generate a range of C values
C_values = np.logspace(-5, 5, num=10)  # 10 values for each γ

# Dictionary to store best C for each γ
best_C_per_gamma = {}
cv_scores_per_gamma = {}

print("Performing double cross-validation for RBF SVM...")

for gamma in gamma_values:
    C_scores = {}

    for C in C_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        mean_score, _ = k_fold_cross_validation(model, X_train, y_train, k=5)
        C_scores[C] = mean_score
        print(f"γ={gamma:.6f}, C={C:.6f}, CV Accuracy={mean_score:.4f}")

    # Select best C_γ for this γ
    best_C = max(C_scores, key=C_scores.get)
    best_C_per_gamma[gamma] = best_C
    cv_scores_per_gamma[gamma] = C_scores[best_C]

# Select the best γ
best_gamma = max(cv_scores_per_gamma, key=cv_scores_per_gamma.get)
best_C_for_best_gamma = best_C_per_gamma[best_gamma]

print(f"\nBest γ: {best_gamma}, Best C: {best_C_for_best_gamma}")

# Save best values
np.save("best_gamma.npy", best_gamma)
np.save("best_C_for_best_gamma.npy", best_C_for_best_gamma)

print("Best γ and C saved for later training.")
