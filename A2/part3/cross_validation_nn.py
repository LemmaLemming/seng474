import sys
sys.path.append("./utils")   
import numpy as np
from sklearn.neural_network import MLPClassifier
from a1_k_fold_cv import k_fold_cross_validation  

# Load preprocessed dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Define Neural Network Model (MLP with 1 Hidden Layer)
model = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", 
                      max_iter=500, random_state=42)

# Perform 5-Fold Cross-Validation
mean_accuracy, scores = k_fold_cross_validation(model, X_train, y_train, k=5)

# Print Results
print(f"Validation Accuracy for each fold: {scores}")
print(f"Average Validation Accuracy across 5-Fold CV: {mean_accuracy:.4f}")
