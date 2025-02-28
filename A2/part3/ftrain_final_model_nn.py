import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier

# Load full training dataset (no validation split this time)
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Define final model (same best configuration)
final_model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", 
                            max_iter=500, random_state=42)

# Train the final model on the full dataset
final_model.fit(X_train, y_train)

# Save the final trained model
joblib.dump(final_model, "final_nn_model.pkl")
print("Final trained model saved as final_nn_model.pkl")
