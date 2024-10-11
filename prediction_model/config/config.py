import pathlib
import os
from datetime import date
import pandas as pd
import prediction_model


PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

FILE_NAME = 'fraud_data.csv'
TEST_FILE = "test_data.csv"

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

#-------------------------------------------------------------------------------------------------------#

TARGET = 'is_fraud'

#Final features used in the model
FEATURES = ['amt', 'city_pop', 'job', 'merch_lat', 'merch_long',
       'owner_age', 'year', 'month', 'day', 'hour', 'min', 'sec',
       'category_entertainment', 'category_food_dining',
       'category_gas_transport', 'category_grocery_net',
       'category_grocery_pos', 'category_health_fitness', 'category_home',
       'category_kids_pets', 'category_misc_net', 'category_misc_pos',
       'category_personal_care', 'category_shopping_net',
       'category_shopping_pos', 'category_travel', 'is_fraud']

PRED_FEATURES = ['amt', 'city_pop', 'job', 'merch_lat', 'merch_long',
       'owner_age', 'year', 'month', 'day', 'hour', 'min', 'sec',
       'category_entertainment', 'category_food_dining',
       'category_gas_transport', 'category_grocery_net',
       'category_grocery_pos', 'category_health_fitness', 'category_home',
       'category_kids_pets', 'category_misc_net', 'category_misc_pos',
       'category_personal_care', 'category_shopping_net',
       'category_shopping_pos', 'category_travel']

#-------------------------------------------------------------------------------------------------------#

NUM_FEATURES = ['amt', 'lat', 'long', 'city_pop','merch_lat','merch_long']

CAT_FEATURES = ['trans_date_trans_time', 'merchant', 'category', 'city', 'state',
                'job', 'dob', 'trans_num', 'is_fraud']

FEATURE_TO_LONG_DATE_TIME = ['trans_date_trans_time']

FEATURE_TO_SHORT_DATE_TIME = ['dob']

TODAY = [pd.to_datetime(date.today(), format='%d-%m-%Y')]

FEATURES_ENGINEERING = ['dob']

FEATURES_EXTRACT = ['year', 'month', 'day', 'hour', 'min', 'sec']

FEATURE_TO_ENCODE = ['category']

FEATURE_TO_CAT_ENCODER = ['job']

FEATURES_TO_DROP = ['lat','long', 'dob', 'trans_date_trans_time', 'merchant',
                 'trans_num', 'city','state']

FEATURES_TO_SCALED = ['amt', 'city_pop', 'job', 'merch_lat', 'merch_long',
       'owner_age', 'year', 'month', 'day', 'hour', 'min', 'sec',
       'category_entertainment', 'category_food_dining',
       'category_gas_transport', 'category_grocery_net',
       'category_grocery_pos', 'category_health_fitness', 'category_home',
       'category_kids_pets', 'category_misc_net', 'category_misc_pos',
       'category_personal_care', 'category_shopping_net',
       'category_shopping_pos', 'category_travel']

GRID_SEARCH = {
     'n_estimators': [80, 90, 100, 110, 120], # Default= 100
     'criterion': ['gini', 'entropy', 'log_loss'], #Default 'gini'
     'min_samples_leaf': [1, 2, 3] #Default 1
}

# Constants
THRESHOLD = 0.85  # Set your performance threshold
MONITORING_INTERVAL = 43200  # Check every twelve hours