import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Define epoch values to test
epoch_values = [10, 50, 100, 200]

results = {}

# Train models for different epochs
for epochs in epoch_values:
    print(f"Training model with {epochs} epochs...")

    # Define model with a single hidden layer of 32 nodes
    model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                          max_iter=epochs, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)  # Training error
    test_error = 1 - accuracy_score(y_test, y_test_pred)  # Test error

    results[epochs] = (train_error, test_error)
    print(f"Epochs: {epochs} - Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

# Plot results
epochs_list = list(results.keys())
train_errors = [results[e][0] for e in epochs_list]
test_errors = [results[e][1] for e in epochs_list]

plt.figure(figsize=(8, 5))
plt.plot(epochs_list, train_errors, marker='o', label="Training Error", linestyle="--")
plt.plot(epochs_list, test_errors, marker='s', label="Test Error", linestyle="--")
plt.xlabel("Number of Training Epochs")
plt.ylabel("Error")
plt.title("Effect of Training Epochs on Training and Test Error")
plt.legend()
plt.grid(True)
plt.show()
