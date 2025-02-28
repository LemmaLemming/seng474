import numpy as np
from utils import mnist_reader

# Load dataset
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Keep only classes 5 (Sandal) and 7 (Sneaker)
train_mask = (y_train == 5) | (y_train == 7)
test_mask = (y_test == 5) | (y_test == 7)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# Relabel: Sandal (5) -> 0, Sneaker (7) -> 1
y_train = np.where(y_train == 5, 0, 1)
y_test = np.where(y_test == 5, 0, 1)

# Ensure images are flattened into vectors of size 784
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Subsample training data: 1000 samples per class
n_train_samples = 1000  
sandal_indices_train = np.where(y_train == 0)[0]
sneaker_indices_train = np.where(y_train == 1)[0]

np.random.seed(42)
sandal_selected_train = np.random.choice(sandal_indices_train, n_train_samples, replace=False)
sneaker_selected_train = np.random.choice(sneaker_indices_train, n_train_samples, replace=False)

selected_train_indices = np.concatenate([sandal_selected_train, sneaker_selected_train])
X_train, y_train = X_train[selected_train_indices], y_train[selected_train_indices]

# Subsample test data: 500 samples per class (1000 total)
n_test_samples = 500  
sandal_indices_test = np.where(y_test == 0)[0]
sneaker_indices_test = np.where(y_test == 1)[0]

sandal_selected_test = np.random.choice(sandal_indices_test, n_test_samples, replace=False)
sneaker_selected_test = np.random.choice(sneaker_indices_test, n_test_samples, replace=False)

selected_test_indices = np.concatenate([sandal_selected_test, sneaker_selected_test])
X_test, y_test = X_test[selected_test_indices], y_test[selected_test_indices]

# Shuffle test data
shuffled_test_indices = np.random.permutation(len(y_test))
X_test, y_test = X_test[shuffled_test_indices], y_test[shuffled_test_indices]

print(f"Final training shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Class distribution in training set: {np.bincount(y_train)}")
print(f"Final test shape: {X_test.shape}, Labels: {y_test.shape}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# Save preprocessed data
np.savez("processed_data.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Processed data saved to 'processed_data.npz'")
