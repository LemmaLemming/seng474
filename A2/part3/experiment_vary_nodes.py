import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Define different models with 32 and 64 nodes
models = {
    "32 Nodes": MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam",
                              max_iter=500, random_state=42),
    "64 Nodes": MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                              max_iter=500, random_state=42)
}

results = {}

# Train both models and evaluate
for name, model in models.items():
    print(f"Training model with {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_error = 1 - accuracy_score(y_train, y_train_pred)  # Training error
    test_error = 1 - accuracy_score(y_test, y_test_pred)  # Test error

    results[name] = (train_error, test_error)
    print(f"{name} - Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

# Plot results
node_sizes = list(results.keys())
train_errors = [results[n][0] for n in node_sizes]
test_errors = [results[n][1] for n in node_sizes]

plt.figure(figsize=(8, 5))
plt.plot(node_sizes, train_errors, marker='o', label="Training Error", linestyle="--")
plt.plot(node_sizes, test_errors, marker='s', label="Test Error", linestyle="--")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Error")
plt.title("Effect of Hidden Layer Size on Training and Test Error")
plt.legend()
plt.grid(True)
plt.show()
