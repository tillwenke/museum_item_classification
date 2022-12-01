# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import wandb

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