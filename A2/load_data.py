import sys
import numpy as np

# Add 'utils' folder to the system path so we can import mnist_reader
sys.path.append("utils")

import mnist_reader

# Load dataset
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# Keep only classes 5 (Sandal) and 7 (Sneaker)
train_mask = (y_train == 5) | (y_train == 7)
test_mask = (y_test == 5) | (y_test == 7)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# Relabel: Sandal (5) -> 0, Sneaker (7) -> 1
y_train = np.where(y_train == 5, 0, 1)
y_test = np.where(y_test == 5, 0, 1)

print(f"Filtered training data: {X_train.shape}, Labels: {y_train.shape}")
print(f"Filtered test data: {X_test.shape}, Labels: {y_test.shape}")

# Ensure images are flattened into vectors of size 784
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened training shape: {X_train.shape}")
print(f"Flattened test shape: {X_test.shape}")

X_train = X_train / 255.0
X_test = X_test / 255.0

# print(f"Final training data shape: {X_train.shape}, Labels: {y_train.shape}")
# print(f"Final test data shape: {X_test.shape}, Labels: {y_test.shape}")
# print(f"Sample label counts: {np.bincount(y_train)}, {np.bincount(y_test)}")
