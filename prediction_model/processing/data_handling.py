from sklearn.model_selection import train_test_split
from prediction_model import config

import os
import sys
import joblib
from pathlib import Path
import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__name__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] #Fixed the white spaces in the names of the columns.
    return _data
    
# Preprocess target variable
def preprocess_target(df, target_variable):
    # Ensure the target variable exists
    if target_variable not in df.columns:
        raise KeyError(f"Column '{target_variable}' not found in DataFrame")
    
    # Filter rows based on the length of target variable values and valid values
    filtered_df = df[df[target_variable].astype(str).str.len() == 1]
    filtered_df = filtered_df[filtered_df[target_variable].isin(["0", "1"])].copy()
    
    # Convert target variable to integer
    filtered_df[target_variable] = filtered_df[target_variable].astype(int)
    
    return filtered_df

#Separate the data into X and y
def separate_data(data):
    X = data.drop(config.TARGET,axis=1)
    y = data[config.TARGET]
    return X,y

#Split the dataset into train and test
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= random_state)
    return X_train, X_test, y_train, y_test

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved unter the name {config.MODEL_NAME}")

#Deserialization
def load_pipeline(pipeline_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded = joblib.load(load_path)
    print(f"Model {config.MODEL_NAME} has been loaded")
    return model_loaded