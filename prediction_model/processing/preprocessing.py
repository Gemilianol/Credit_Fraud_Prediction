from sklearn.base import BaseEstimator, TransformerMixin

from pathlib import Path
import os
import sys
from prediction_model import config

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

import numpy as np
import pandas as pd

class DropColumns(TransformerMixin,BaseEstimator):
    def __init__(self, variables_to_drop: list):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop(columns = self.variables_to_drop)
        return X

class LongDateColumns(BaseEstimator,TransformerMixin):
    def __init__(self, long_variables_date):
        self.long_variables_date = long_variables_date
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.long_variables_date] = pd.to_datetime(X[self.long_variables_date].stack(), format='%d-%m-%Y %H:%M').unstack()
        return X 

class ShortDateColumns(BaseEstimator,TransformerMixin):
    def __init__(self, short_variables_date):
        self.short_variables_date = short_variables_date
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.short_variables_date] = pd.to_datetime(X[self.short_variables_date].stack(), format='%d-%m-%Y').unstack()
        return X 
    
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, dob, age='owner_age'):
        self.today = config.TODAY
        self.age = age
        self.dob = dob
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.age] = ((self.today - X[self.dob]) / np.timedelta64(1, 'W') / 52).astype(int)
        return X

class FeaturesExtract(BaseEstimator, TransformerMixin):
    '''
    X[self.date] returns a DataFrame with a single column.
    .dt is a method of pd.Series so you need to squeeze('columns')
    '''
    def __init__(self, date, year='year', month='month', day='day',
                 hour='hour', minutes='min', sec='sec'):
        self.date = date
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minutes = minutes
        self.sec = sec
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.year] = X[self.date].squeeze('columns').dt.year
        X[self.month] = X[self.date].squeeze('columns').dt.month
        X[self.day] = X[self.date].squeeze('columns').dt.day
        X[self.hour] = X[self.date].squeeze('columns').dt.hour
        X[self.minutes] = X[self.date].squeeze('columns').dt.minute
        X[self.sec] = X[self.date].squeeze('columns').dt.second
        return X

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, variable):
        self.variable = variable
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.get_dummies(X, columns=self.variable, dtype='int')
        return X