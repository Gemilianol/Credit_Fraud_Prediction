import pipeline as pipe
from prediction_model import config
from processing.data_handling import load_dataset, save_pipeline, separate_data, split_data, preprocess_target

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__name__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

import prediction_model.processing.preprocessing as pp
from category_encoders.cat_boost import CatBoostEncoder

def perform_training():
    """
    This function is the responsable for run the training step
    """
    dataset = load_dataset(config.FILE_NAME)
    
    #First convert all the data in a proper way (Only data format - No Data Leakage)
    dataset = preprocess_target(dataset, config.TARGET)
    
    dataset = pipe.features_handling_pipeline.transform(dataset)
    
    X, y = separate_data(dataset)
    
    X_train, X_test, y_train, y_test = split_data(X,y)
    
    # The issue here is that, if I put the transformer into a pipeline, never riches the y_train
    # so I need to work with it separately.
    # In a pipeline, y_train is only reaches for the predictor (or the last transformer)
    
    encoder = CatBoostEncoder(cols=config.FEATURE_TO_CAT_ENCODER)
    
    X_train = encoder.fit_transform(X_train, y_train)
    
    X_test = encoder.transform(X_test)
    
    test_data = X_test.copy()
    
    test_data[config.TARGET] = y_test
    
    test_data.to_csv(os.path.join(config.DATAPATH,config.TEST_FILE), index=False)
    
    pipe.classification_pipeline.fit(X_train, y_train)

    save_pipeline(pipe.classification_pipeline)

if __name__ == '__main__':
    perform_training()