import pickle
import pandas as pd

models = {}

models['column'] = pickle.load()
models['text'] = pickle.load()

data = pd.read_csv('data/inference/AM_ETMM.csv')
