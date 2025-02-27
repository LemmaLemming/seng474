all: data preprocess tune train

data:
    python utils/load_data.py

preprocess: data
    cd utils && python generate_c_values.py

tune: preprocess
    python utils/cross_validation.py

train: tune
    python utils/train_final_model.py
