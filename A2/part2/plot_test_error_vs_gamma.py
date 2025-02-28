import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Load all γ and best Cγ values
with open("best_C_per_gamma.json", "r") as f:
    best_C_per_gamma = json.load(f)

gamma_values = np.array(list(best_C_per_gamma.keys()), dtype=float)
test_errors = []

# Evaluate test error for each gamma value
for gamma in gamma_values:
    C = best_C_per_gamma[str(gamma)]  # Retrieve best Cγ for this γ
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(X_train, y_train)
    test_error = 1 - accuracy_score(y_test, model.predict(X_test))
    test_errors.append(test_error)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, test_errors, marker='o', linestyle='-', label="Test Error")
plt.xscale("log")
plt.xlabel("Gamma (γ)")
plt.ylabel("Test Error")
plt.title("Test Error vs. Gamma for Tuned SVMs")
plt.legend()
plt.grid(True)
plt.show()
