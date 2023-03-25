import pandas as pd
import numpy as np
import os

from sklearn.naive_bayes import MultinomialNB, ComplementNB

# can take some time to run - import carefully

if os.popen('hostname').read() != 'till\n':
    curie = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/curie.csv', index_col='id', dtype={'type': str})
    """
    train_curie = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/train_curie.csv', index_col='id', dtype={'type': str})
    val_curie = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/val_curie.csv', index_col='id', dtype={'type': str})
    test_curie = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/test_curie.csv', index_col='id', dtype={'type': str})

    train_bow = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/train_bow.csv', index_col='id', dtype={'type': str})
    val_bow = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/val_bow.csv', index_col='id', dtype={'type': str})
    test_bow = pd.read_csv('/gpfs/space/home/till/museum/museum_item_classification/data/text_embeddings/test_bow.csv', index_col='id', dtype={'type': str})
    """
    
    with open('/gpfs/space/home/till/museum/museum_item_classification/data/general/estonian-stopwords.txt') as file:
        lines = [line.rstrip() for line in file]
    stopwords_est = lines

else:
    curie = pd.read_csv('data/text_embeddings/curie.csv', index_col='id', dtype={'type': str})
    train_curie = pd.read_csv('data/text_embeddings/train_curie.csv', index_col='id', dtype={'type': str})
    val_curie = pd.read_csv('data/text_embeddings/val_curie.csv', index_col='id', dtype={'type': str})
    test_curie = pd.read_csv('data/text_embeddings/test_curie.csv', index_col='id', dtype={'type': str})

    train_bow = pd.read_csv('data/text_embeddings/train_bow.csv', index_col='id', dtype={'type': str})
    val_bow = pd.read_csv('data/text_embeddings/val_bow.csv', index_col='id', dtype={'type': str})
    test_bow = pd.read_csv('data/text_embeddings/test_bow.csv', index_col='id', dtype={'type': str})

    with open('data/general/estonian-stopwords.txt') as file:
        lines = [line.rstrip() for line in file]
    stopwords_est = lines

# categorize and resort types
types = [
'sculpture',\
'bag', 'suit', 'doll', 'sheet/linen', 'dish/vessel','jewel', 'tape/ribbon',\
'country',\
'paper','book','magazines', 'album', 'newspaper', 'folder/booklet',\
'invitation',  'calendar',\
'audio recording', 'telegram',\
'packaging', 'crate/box',\
'printed notes', 'small print',\
'seal', 'seal/imprint',\
'letter','letter of honor/honorary address',\
'postcard', 'photo, postcard', 'letter, postcard',\
'manuscript','script, song/vocal music', 'music sheet', 'musical instrument', 'manuscript, musical composition', 'manuscript, sheet music',\
'medal', 'coin', 'label/sign',\
'poster','plan','notes', 'document', 'certificate',\
'graphics', 'drawing', 'design/drawing/sketch','caricature','slide',\
'archaeological find',\
'photo', 'photo negative', 'photographic negative, photographic negative', 'photographic material','digital image'
]
