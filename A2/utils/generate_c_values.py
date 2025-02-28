import numpy as np

# Define parameters for C value generation
C0 = 0.0001  
beta = 10   
num_C_values = 10  


C_values = np.array([C0 * (beta ** i) for i in range(num_C_values)])

np.save("c_values.npy", C_values)

print("Generated C values:", C_values)
print("C values saved to 'c_values.npy'")
