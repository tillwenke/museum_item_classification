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

import xgboost as xgb
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
# utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

data = pd.read_csv('data/train_curie.csv', index_col='id', dtype={'type': str})
data['curie_similarity'] = data.curie_similarity.apply(eval).apply(np.array)

features = np.array(list(data.curie_similarity.values))
labels = data.type
# at least xgboost cannot deal with string labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
labels = label_encoder.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size = 0.3,
    random_state=0
)

clf = TabNetClassifier()
clf.fit(
  X_train, y_train,
  eval_set=[(X_test, y_test)], patience=0, max_epochs=100
)

preds = clf.predict(X_test.values)
probs = clf.predict_proba(X_test.values)
val_acc = accuracy_score(y_test, preds)
print(val_acc)

clf.save_model('models/nlp/with_nn')