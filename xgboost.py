# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import re 
from math import isnan
import wandb
import random
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
# dtype={'type': str} prevents being confused with data type for large data sets
train = pd.read_csv('data/train.csv', index_col='id', dtype={'type': str})
test = pd.read_csv('data/test.csv', index_col='id', dtype={'type': str})
train_translated = pd.read_csv('data/train_translated.csv', dtype={'type': str})
test_translated = pd.read_csv('data/test_translated.csv', index_col='id', dtype={'type': str})
combined_data = pd.read_csv('data/combined_data.csv', index_col='id', dtype={'type': str})
combined_data_translated = pd.read_csv('data/combined_data_translated.csv', index_col='id', dtype={'type': str})
combined_data_fully_translated = pd.read_csv('data/combined_data_fully_translated.csv', index_col='id', dtype={'type': str})
prep = pd.read_csv('data/prep.csv', index_col='id', dtype={'type': str})
test_prep = pd.read_csv('data/test_prepared.csv', index_col='id', dtype={'type': str})
train_prep = pd.read_csv('data/train_prepared.csv', index_col='id', dtype={'type': str})


# xg boost

import xgboost as xgb
from xgboost import XGBClassifier
# utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data = train_prep.copy()
features = data.drop('type', axis=1)
labels = data.type
# at least xgboost cannot deal with string labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
labels = label_encoder.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.0001, random_state=0)
bst = XGBClassifier(random_state=0)
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)
val_acc = accuracy_score(y_test, preds)
val_acc
bst.save_model('models/xgboost_full.json')