import numpy as np

# Define parameters for C value generation
C0 = 0.0001  # Initial value of C
beta = 10    # Multiplication factor (logarithmic spacing)
num_C_values = 10  # Number of values to generate

# Generate C values: C0, beta*C0, beta^2*C0, ..., beta^(num_C_values-1) * C0
C_values = np.array([C0 * (beta ** i) for i in range(num_C_values)])

# Save the generated values to a file
np.save("c_values.npy", C_values)

print("Generated C values:", C_values)
print("C values saved to 'c_values.npy'")
