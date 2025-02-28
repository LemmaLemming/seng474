import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load preprocessed dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Split data for early stopping validation (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", 
                      max_iter=1, warm_start=True, random_state=42)

# Early stopping parameters
best_val_loss = float("inf")
patience = 5
no_improve_epochs = 0
best_model = None

# Training loop with early stopping
for epoch in range(200):  # Max 200 epochs
    model.fit(X_train, y_train)  # Train for one epoch

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_loss = 1 - accuracy_score(y_val, y_val_pred)  # Loss = 1 - accuracy

    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model  # Save best model
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if no_improve_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Save the final best model
import joblib
joblib.dump(best_model, "best_nn_model.pkl")
print("Best model saved as best_nn_model.pkl")
