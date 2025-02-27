import numpy as np

def k_fold_cross_validation(model, X, y, k=5, scoring='accuracy'):
    """
    Perform k-fold cross-validation manually (without sklearn).
    
    Parameters:
    - model: A machine learning model (must have fit and predict methods).
    - X: Feature matrix (numpy array).
    - y: Target labels (numpy array).
    - k: Number of folds (default is 5).
    - scoring: Metric to evaluate model performance (default: 'accuracy').

    Returns:
    - mean_score: Average score across k folds.
    - scores: List of individual fold scores.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.seed(42)  # Ensure reproducibility
    np.random.shuffle(indices)  # Shuffle indices before splitting

    fold_sizes = np.full(k, n_samples // k, dtype=int)  # Equal-sized folds
    fold_sizes[:n_samples % k] += 1  # Distribute remaining samples

    current = 0
    scores = []

    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate((indices[:current], indices[current + fold_size:]))
        
        current += fold_size
        
        # Split the data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        # Compute accuracy
        score = np.mean(predictions == y_val)
        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score, scores
