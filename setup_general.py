# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import wandb
import pickle
import re
import helpers

import xgboost as xgb
from xgboost import XGBClassifier
# utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

import openai
import time
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier

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
low = pd.read_csv('data/prep_low_thres.csv', index_col='id', dtype={'type': str})
high = pd.read_csv('data/prep_high_thres.csv', index_col='id', dtype={'type': str})

train_prepROS = pd.read_csv('data/train_prepROS.csv')
train_prepSMOTE = pd.read_csv('data/train_prepSMOTE100.csv')

train_text = pd.read_csv('data/train_text.csv', index_col='id', dtype={'type': str})
test_text = pd.read_csv('data/test_text.csv', index_col='id', dtype={'type': str})

type_lookup = pd.read_csv('data/type_lookup.csv')