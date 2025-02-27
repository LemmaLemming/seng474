import numpy as np
from sklearn.svm import SVC
import joblib  # To save the model

# Load preprocessed data
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Best C from cross-validation
best_C = 0.1  # Replace with the actual best C from your results

# Train final SVM model
final_model = SVC(kernel="linear", C=best_C)
final_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(final_model, "best_svm_model.pkl")

print(f"Final model trained with C={best_C} and saved as 'best_svm_model.pkl'")
