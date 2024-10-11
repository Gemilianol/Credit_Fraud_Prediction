from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__name__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model import config
import prediction_model.processing.preprocessing as pp
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.preprocessing import StandardScaler

features_handling_pipeline = Pipeline(
    [('LongDates', pp.LongDateColumns(config.FEATURE_TO_LONG_DATE_TIME)),
    ('ShortDates', pp.ShortDateColumns(config.FEATURE_TO_SHORT_DATE_TIME)),
    ('FeaturesExtract', pp.FeaturesExtract(config.FEATURE_TO_LONG_DATE_TIME)),
    ('FeatureEngineering', pp.FeatureEngineering(config.FEATURES_ENGINEERING)),
    ('OHE', pp.OneHotEncoding(config.FEATURE_TO_ENCODE)),
    ('DropFeatures', pp.DropColumns(config.FEATURES_TO_DROP))]
    )

classification_pipeline = Pipeline(
    [("Scaler", StandardScaler()),
    ("RandomForest", RandomForestClassifier())
     ]
)

retrain_pipeline = RandomizedSearchCV(
    classification_pipeline,
    param_distributions= config.GRID_SEARCH,
    n_iter=100,
    cv=5,      
    verbose=1,
    random_state=42,
    n_jobs=-1    
)