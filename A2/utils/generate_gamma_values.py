import numpy as np

# Generate γ values logarithmically spaced from 10⁻⁵ to 10⁵
gamma_values = np.logspace(-5, 5, num=20)  # 20 values

# Save the γ values
np.save("gamma_values.npy", gamma_values)

print(f"Generated γ values: {gamma_values}")
print("Gamma values saved to 'gamma_values.npy'")
