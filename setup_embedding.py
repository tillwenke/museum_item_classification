import pandas as pd
import numpy as np

train_curie = pd.read_csv('data/train_curie.csv', index_col='id', dtype={'type': str})
test_curie = pd.read_csv('data/test_curie.csv', index_col='id', dtype={'type': str})
train_curie['curie_similarity'] = train_curie.curie_similarity.apply(eval).apply(np.array)
test_curie['curie_similarity'] = test_curie.curie_similarity.apply(eval).apply(np.array)