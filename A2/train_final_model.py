import numpy as np
from sklearn.svm import SVC
import joblib  # To save the model
import os

# Load preprocessed data
data = np.load("utils/processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load best C from file
best_C = np.load("utils/best_C.npy")
print(f"Loaded best C: {best_C}")

# Train final SVM model using best C
final_model = SVC(kernel="linear", C=best_C)
final_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(final_model, "utils/best_svm_model.pkl")
print(f"Final model trained with C={best_C} and saved as 'utils/best_svm_model.pkl'")
