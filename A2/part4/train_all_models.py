import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load best hyperparameters
best_C = np.load("utils/best_C.npy").item()  # Best C for Linear SVM
best_gamma = np.load("best_gamma.npy").item()  # Best gamma for Gaussian SVM
best_C_gaussian = np.load("best_C_for_best_gamma.npy").item()  # Best C for Gaussian SVM

# ---- Train Linear SVM ----
print("Training Linear SVM...")
linear_svm = SVC(kernel="linear", C=best_C, random_state=42)
linear_svm.fit(X_train, y_train)
joblib.dump(linear_svm, "linear_svm.pkl")
print("Linear SVM saved as linear_svm.pkl")

# ---- Train Gaussian Kernel SVM ----
print("Training Gaussian Kernel SVM...")
gaussian_svm = SVC(kernel="rbf", C=best_C_gaussian, gamma=best_gamma, random_state=42)
gaussian_svm.fit(X_train, y_train)
joblib.dump(gaussian_svm, "gaussian_svm.pkl")
print("Gaussian Kernel SVM saved as gaussian_svm.pkl")

# ---- Train Neural Network (if needed) ----
print("Retraining Neural Network on full dataset...")
best_nn_model = joblib.load("final_nn_model.pkl")  # Load pre-trained model
best_nn_model.fit(X_train, y_train)
joblib.dump(best_nn_model, "trained_nn_model.pkl")
print("Neural Network saved as trained_nn_model.pkl")

print("Step 1 complete: All models trained and saved.")
