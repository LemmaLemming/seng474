import numpy as np
from sklearn.svm import SVC
from joblib import dump

# Load dataset
data = np.load("processed_data.npz")
X_train, y_train = data["X_train"], data["y_train"]

# Load best hyperparameters and convert to float
best_gamma = float(np.load("best_gamma.npy"))
best_C = float(np.load("best_C_for_best_gamma.npy"))

# Train final SVM model
final_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
final_model.fit(X_train, y_train)

# Save the trained model
dump(final_model, "final_svm_model.joblib")

print(f"Final model trained with γ={best_gamma}, C={best_C} and saved as 'final_svm_model.joblib'")
