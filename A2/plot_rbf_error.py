import numpy as np
import matplotlib.pyplot as plt

# Load stored results
gamma_values = np.load("utils/gamma_values.npy")
training_errors = np.load("utils/training_errors.npy")
test_errors = np.load("utils/test_errors.npy")

# Plot error vs. γ
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, training_errors, label="Training Error", marker="o", linestyle='-')
plt.plot(gamma_values, test_errors, label="Test Error", marker="s", linestyle='-')
plt.xscale("log")
plt.xlabel("Gamma (γ)")
plt.ylabel("Error Rate")
plt.title("Training and Test Error vs. Gamma (γ)")
plt.legend()
plt.grid()
plt.savefig("error_vs_gamma_plot.png")
plt.show()
