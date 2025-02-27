import numpy as np
from sklearn.svm import SVC
import joblib  # Save model

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load best γ and C_γ
best_gamma = np.load("best_gamma.npy")
best_C = np.load("best_C_for_best_gamma.npy")

print(f"Training final SVM with γ={best_gamma}, C={best_C}...")

# Train final model
final_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
final_model.fit(X_train, y_train)

# Save trained model
joblib.dump(final_model, "best_svm_rbf.pkl")
print("Final RBF SVM model saved as 'best_svm_rbf.pkl'.")
