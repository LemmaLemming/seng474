# Run data preprocessing
python utils/load_data.py

# Move into utils directory to generate C values
cd utils
python generate_c_values.py
cd ..

# Run cross-validation
python utils/cross_validation.py

# Train the final model
python utils/train_final_model.py

Write-Host "Pipeline execution complete!"
