import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load("processed_data.npz")
X_test, y_test = data["X_test"], data["y_test"]

# Load trained models
linear_svm = joblib.load("linear_svm.pkl")
gaussian_svm = joblib.load("gaussian_svm.pkl")
neural_network = joblib.load("trained_nn_model.pkl")

# Evalute
def compute_test_error(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_pred)
    return test_error

# Compute errors
linear_svm_error = compute_test_error(linear_svm, X_test, y_test)
gaussian_svm_error = compute_test_error(gaussian_svm, X_test, y_test)
nn_error = compute_test_error(neural_network, X_test, y_test)

# Print results
print(f"Test Error - Linear SVM: {linear_svm_error:.4f}")
print(f"Test Error - Gaussian SVM: {gaussian_svm_error:.4f}")
print(f"Test Error - Neural Network: {nn_error:.4f}")

# Save results
np.savez("test_errors.npz", linear_svm=linear_svm_error, gaussian_svm=gaussian_svm_error, nn=nn_error)

print("Step 2 complete: Test errors computed and saved.")
