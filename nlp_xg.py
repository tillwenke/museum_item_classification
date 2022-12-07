
# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import re 
from math import isnan
import wandb
from xgboost import XGBClassifier

# utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

df = pd.read_csv('data/train_curie.csv', index_col='id', dtype={'type': str})
df['curie_similarity'] = df.curie_similarity.apply(eval).apply(np.array)

features = list(df.curie_similarity.values)
labels = df.type
# at least xgboost cannot deal with string labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
labels = label_encoder.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

bst = XGBClassifier(random_state=0)
print('run')
# fit model
bst.fit(X_train, y_train)
print('ran')
# make predictions
preds = bst.predict(X_test)
val_acc = accuracy_score(y_test, preds)

y_pred = bst.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)

print(train_acc, val_acc)

bst.save_model('models/nlp/xgboost.json')