import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the full training and test dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Generate a larger set of C values for broader analysis
C_values = np.logspace(-5, 5, num=20)  # 20 values from 10⁻⁵ to 10⁵

training_errors = []
test_errors = []

print("Training models for different C values...")

for C in C_values:
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)
    
    # Compute training error
    y_train_pred = model.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    training_errors.append(train_error)

    # Compute test error
    y_test_pred = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)

    print(f"C={C:.6f}, Training Error={train_error:.4f}, Test Error={test_error:.4f}")

# Plot Training & Test Error vs. C
plt.figure(figsize=(8, 6))
plt.plot(C_values, training_errors, label="Training Error", marker="o", linestyle='-')
plt.plot(C_values, test_errors, label="Test Error", marker="s", linestyle='-')
plt.xscale("log")  # Log scale for better visualization
plt.xlabel("C (Regularization Parameter)")
plt.ylabel("Error Rate")
plt.title("Training and Test Error vs. C")
plt.legend()
plt.grid()
plt.savefig("utils/error_vs_C_plot.png")  # Save plot
plt.show()

print("Plot saved as 'utils/error_vs_C_plot.png'")
