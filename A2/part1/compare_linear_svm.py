from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Load best regularization parameter
best_C = float(np.load("best_C_for_best_gamma.npy"))

# Train Linear SVM
linear_svm = SVC(kernel="linear", C=best_C)
linear_svm.fit(X_train, y_train)

# Compute test error for linear SVM
linear_test_error = 1 - accuracy_score(y_test, linear_svm.predict(X_test))

print(f"Linear SVM Test Error: {linear_test_error:.4f}")
