import numpy as np
from sklearn.metrics import accuracy_score
from joblib import load

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Load trained model
final_model = load("final_svm_model.joblib")

# Compute training and test accuracy
train_accuracy = accuracy_score(y_train, final_model.predict(X_train))
test_accuracy = accuracy_score(y_test, final_model.predict(X_test))

print(f"Final Model Training Accuracy: {train_accuracy:.4f}")
print(f"Final Model Test Accuracy: {test_accuracy:.4f}")
