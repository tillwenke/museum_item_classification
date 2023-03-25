import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from inference_utils import *

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
models = {}

dir = 'models/column/'
models['column'] = pickle.load(file = open(dir + 'best_model_rfc_column_2023-03-24_16-04-33_0.6658043781807724.pkl', 'rb'))
dir = 'models/text/'
models['text'] = pickle.load(file = open(dir + 'best_model_rfc_text_2023-03-24_17-12-20_0.5555078639169283.pkl', 'rb'))

data = {}
data['column'] = pd.read_csv('data/inference/AM_ETMM_column.csv', index_col='id')
data['text'] = text_to_bow(path='data/inference/AM_ETMM_text.csv')
results = pd.DataFrame()
results['id'] = data['column'].index
results.set_index('id', inplace=True)

for key, model in models.items():
    results[key] = [[-1]] * len(results)

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

for key, model in models.items():
    current_data = data[key]
    model_features = model.feature_names_in_
    data_features = current_data.columns
    print(key, len(model_features), 'features in model', len(data_features, 'features in data')

    # align features
    model_not_data = [feature for feature inf model_features if feature not in data_features]
    current_data[model_not_data] = 0
    intersection = [feature for feature in current_data.columns if feature in model_features]
    current_data = current_data[intersection]
    current_data = current_data[model_features]
    data[key] = current_data   

for key in models.keys():
    for i,item in enumerate(models[key].predict_proba(data[key])):
        id = data[key].iloc[i].name
        results[key].loc[id] = np.array(item)
# !!! make sure models have seen same labels
results['prediction'], results['certainty'] = zip(*results.apply(lambda row: vote([row.column, row.text]), axis=1))
type_lookup = pd.read_csv('data/inference/type_lookup.csv', index_col=0)
results.prediction = results.prediction.replace(type_lookup.id_est.to_list(), type_lookup.estonian.to_list())

# replace if label already in text
a = results.prediction.copy().tolist()

raw = pd.read_csv("data/inference/AM_ETMM_raw.csv", index_col='id')
raw_labels = raw.apply(extract_label_from_comment, axis=1)

results.submission = replace_predictions(raw_labels, results.prediction)

b = results.submission.copy().tolist()

count = 0
for i in range(len(a)):
    if a[i] != b[i]:
        count += 1
        #print(a[i], b[i])

print(count, 'replaced by label from text')

# submissions are overwritten predictions
raw['submission'],raw['prediction'], raw['certainty'] = results.submission, results.prediction, results.certainty
raw.to_csv('data/inference/AM_ETMM_results.csv')