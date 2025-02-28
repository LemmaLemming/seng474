import numpy as np
import scipy.stats as stats

# Load test errors
errors = np.load("test_errors.npz")
linear_svm_error = errors["linear_svm"]
gaussian_svm_error = errors["gaussian_svm"]
nn_error = errors["nn"]

# Test set size
n_test_samples = 1000  # Make sure this matches your test set size

# Function to compute confidence interval
def compute_confidence_interval(error, n_samples, confidence=0.95):
    std_error = np.sqrt((error * (1 - error)) / n_samples)  # Standard error
    margin = stats.norm.ppf((1 + confidence) / 2) * std_error  # Margin of error
    return error - margin, error + margin  # Lower & upper bounds

# Compute confidence intervals
ci_linear = compute_confidence_interval(linear_svm_error, n_test_samples)
ci_gaussian = compute_confidence_interval(gaussian_svm_error, n_test_samples)
ci_nn = compute_confidence_interval(nn_error, n_test_samples)

# Print results
print(f"Confidence Interval (95%) - Linear SVM: {ci_linear}")
print(f"Confidence Interval (95%) - Gaussian SVM: {ci_gaussian}")
print(f"Confidence Interval (95%) - Neural Network: {ci_nn}")

# Save results
np.savez("confidence_intervals.npz", linear=ci_linear, gaussian=ci_gaussian, nn=ci_nn)
print("Step 3 complete: Confidence intervals computed and saved.")
