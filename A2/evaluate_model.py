import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Load preprocessed data
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Load the trained SVM model
model = joblib.load("best_svm_model.pkl")

# Predict on training data
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict on test data
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Save evaluation results
eval_file = "evaluation_results.txt"
with open(eval_file, "w") as f:
    f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

# Print results
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Evaluation results saved to '{eval_file}'")
