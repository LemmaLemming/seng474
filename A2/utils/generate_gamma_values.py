import numpy as np

def generate_hyperparameters():
    # Log-spaced γ (bandwidth parameter) with a broader range
    gamma_values = np.logspace(-5, 2, num=10)  # 10 values from 10^-5 to 10^2
    
    # Log-spaced C (regularization parameter) with a standard range
    C_values = np.logspace(-3, 3, num=10)  # 10 values from 10^-3 to 10^3
    
    # Save as .npy files
    np.save("gamma_values.npy", gamma_values)
    np.save("C_values.npy", C_values)
    
    print("Generated and saved gamma_values.npy and C_values.npy")

if __name__ == "__main__":
    generate_hyperparameters()