from pathlib import Path
import os
import sys
import numpy as np
import pytest

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__name__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, separate_data, load_pipeline
from prediction_model.predict import generate_predictions

classification_pipeline = load_pipeline(config.MODEL_NAME)

#@pytest.fixture just indicates that the return value of the decorated function 
# will be use as parameter in other test function. (The decorated function will not run as a test).
@pytest.fixture
def single_prediction():
    test_data = load_dataset(config.TEST_FILE)
    X,y = separate_data(test_data)
    pred = classification_pipeline.predict(X)
    return pred

#The name functions after the decorator need to start with test_*: 
def test_single_prediction_not_none(single_prediction):
    assert single_prediction is not None
    
def test_single_prediction_str_type(single_prediction):
    print(f'single_prediction[0]: {single_prediction[0]}, type: {type(single_prediction[0])}')
    assert isinstance(single_prediction[0],np.int64)