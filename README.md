# Classification of items in Estonian Museums
Part of this [kaggle challenge](https://www.kaggle.com/competitions/caps-in-museums-data-part-2)

[Column discovery](https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing)

# Getting started
Embed the cloned project in a [PyVenv](https://docs.python.org/3/library/venv.html) (or conda env if you prefer) and install the requiremnts listed in requirements.txt

Due to large files you have to get trained models and text embeddings from [here](https://www.dropbox.com/sh/a8nhnkf63um3r29/AADpV1nNK4aR2CJBiB7YONB9a?dl=0) and paste them respectively in models/ and data/text_embeddings


# Executing
Mostly you will see Jupyter Notebooks. Especially during data preparation we went in baby steps therefore you will often notice something similar to:

```
data = dataset.copy
"do something with data"
data.to_csv('dataset.csv)
```
execute this block - to have the changes present in the current notebook you will have to read the files in again via "from setup_general import *" and you can proceed in the notebook from top to bottom.

# How to use:

Run the notebooks in the following order:

### Data discovery:

data_discovery

### Data Preparation:

combine_test_and_train
translation
whitespaces
unit_sizes
split_content_feature
formatting_1
formatting_2
hot_encoding_prep - upto Re-hot encoding 
adding_and_deleting_columns
prep_final

To adjusting thresholds or to remove extra columns later:
hot_encoding_prep - from Re-hot encoding 
adding_and_deleting_columns


### Modelling: 

prepare for models:
rebalancing

Running models - here order in not as important 

random_forest
random_forest_balanced
crossval_randomforest
neural_networks
nlp
predictions
xgboost_tries
xgboost_tries_balanced
voter

The details of these notebooks are found in README_data_prep and README_modelling respectively 
