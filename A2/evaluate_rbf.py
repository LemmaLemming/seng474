import numpy as np
from sklearn.svm import SVC
from utils.k_fold_cv import k_fold_cross_validation  # Import custom k-fold CV function

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Load gamma values
gamma_values = np.load("utils/gamma_values.npy")

# Generate a range of C values
C_values = np.logspace(-5, 5, num=10)  # 10 values for each γ

# Dictionaries to store best C and errors for each γ
best_C_per_gamma = {}
cv_scores_per_gamma = {}
training_errors = []
test_errors = []

print("Performing double cross-validation for RBF SVM...")

for gamma in gamma_values:
    C_scores = {}

    for C in C_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        mean_score, _ = k_fold_cross_validation(model, X_train, y_train, k=5)
        C_scores[C] = mean_score
        print(f"γ={gamma:.6f}, C={C:.6f}, CV Accuracy={mean_score:.4f}")

    # Select best C for this γ
    best_C = max(C_scores, key=C_scores.get)
    best_C_per_gamma[gamma] = best_C
    cv_scores_per_gamma[gamma] = C_scores[best_C]

    # Train final model on full training set with best C_γ
    final_model = SVC(kernel="rbf", C=best_C, gamma=gamma)
    final_model.fit(X_train, y_train)

    # Compute training error
    y_train_pred = final_model.predict(X_train)
    train_error = 1 - np.mean(y_train_pred == y_train)
    training_errors.append(train_error)

    # Compute test error
    y_test_pred = final_model.predict(X_test)
    test_error = 1 - np.mean(y_test_pred == y_test)
    test_errors.append(test_error)

    print(f"γ={gamma:.6f}, Best C={best_C:.6f}, Training Error={train_error:.4f}, Test Error={test_error:.4f}")

# Select the best γ
best_gamma = max(cv_scores_per_gamma, key=cv_scores_per_gamma.get)
best_C_for_best_gamma = best_C_per_gamma[best_gamma]

print(f"\nBest γ: {best_gamma}, Best C: {best_C_for_best_gamma}")

# Save best values
np.save("utils/best_gamma.npy", best_gamma)
np.save("utils/best_C_for_best_gamma.npy", best_C_for_best_gamma)

# Save training and test errors for plotting
np.save("utils/training_errors.npy", np.array(training_errors))
np.save("utils/test_errors.npy", np.array(test_errors))

print("Best γ and C saved for later training.")
print("Training and test errors saved for plotting.")
