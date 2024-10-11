import pandas as pd
import numpy as np
import time
from pathlib import Path
import os
import sys
import joblib
import random

from sklearn import metrics
import pipeline as pipe
from prediction_model import config
from processing.data_handling import load_dataset, save_pipeline, separate_data, split_data, preprocess_target
from prediction_model.processing.data_handling import load_pipeline, load_dataset, separate_data
import prediction_model.processing.preprocessing as pp
from category_encoders.cat_boost import CatBoostEncoder

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__name__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

classification_pipeline = load_pipeline(config.MODEL_NAME)


def evaluate_model():
    test_data = load_dataset(config.TEST_FILE)
    X, y = separate_data(test_data)
    preds = classification_pipeline.predict(X)
    
    #ROC and AUC metric:
    fpr, tpr, thresholds = metrics.roc_curve(y, preds)
    performance = np.round(metrics.auc(fpr, tpr),4)
    
    print(f"Current Test Performance: {performance}")
    
    return performance

def trigger_training_pipeline():
    """
    This function is reponsable for monitoring the performance of the model and,
    if it is neccesary, retrain it.
    """    
    print("Triggering the training pipeline...")
    
    #At this time, no make sense becuase the dataset it's static. It's just for check if the script runs well. 
    dataset = load_dataset(config.FILE_NAME)
    
    #First convert all the data in a proper way (Only data format - No Data Leakage)
    dataset = preprocess_target(dataset, config.TARGET)
    
    dataset = pipe.features_handling_pipeline.transform(dataset)
    
    X, y = separate_data(dataset)
    
    #Here I'll change the seed in order to generate different test set if the model need to be retrain. 
    X_train, X_test, y_train, y_test = split_data(X,y, random_state=random.randint(1,100000))
    
    encoder = CatBoostEncoder(cols=config.FEATURE_TO_CAT_ENCODER)
    
    X_train = encoder.fit_transform(X_train, y_train)
    
    X_test = encoder.transform(X_test)
    
    test_data = X_test.copy()
    
    test_data[config.TARGET] = y_test
    
    test_data.to_csv(os.path.join(config.DATAPATH,config.TEST_FILE), index=False)
    
    pipe.retrain_pipeline.fit(X_train, y_train)

    save_pipeline(pipe.retrain_pipeline)

def monitor_and_trigger():
    while True:
        performance = evaluate_model()
        
        if performance < config.THRESHOLD:
            trigger_training_pipeline()
        
        time.sleep(config.MONITORING_INTERVAL)

if __name__ == "__main__":
    monitor_and_trigger()
