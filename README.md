# Classification of items in Estonian Museums
Part of this [kaggle challenge](https://www.kaggle.com/competitions/caps-in-museums-data-part-2)

In [Column Discovery](https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing) our findings and resulted feature engineering tasks per feature are captured.

# Getting started
Embed the cloned project in a [PyVenv](https://docs.python.org/3/library/venv.html) (or conda env if you prefer) and install the requiremnts listed in requirements.txt

Due to large files you have to get trained models and text embeddings from [here](https://drive.google.com/drive/folders/1ZKOynMoLZd0R_0Q8kB-AwELzFwo50atb?usp=sharing) and paste them respectively in models/ and data/text_embeddings


# Executing
Mostly you will see Jupyter Notebooks. Especially during data preparation we went in baby steps therefore you will often notice something similar to:

```
data = dataset.copy
"do something with data"
data.to_csv('dataset.csv)
```

execute this block - to have the changes present in the current notebook you will have to read the files in again via `from setup_general import *` and you can proceed in the notebook from top to bottom.

We set settings so that Notebook outputs are fully displayed for investigation purposes. But this makes it necessary to limit what you want to see when displaying variables containing large amount of data, so for example rather do `prep.head()`than `prep`.

# How to use:

Run the notebooks in the following order:

### Data discovery:

data_discovery

### Data Preparation:
For more detailed information see: README_data_prep.md

- prep_combine_test_and_train
- prep_translation
- prep_whitespace_deletion
- prep_units_sizes
- prep_split_feature_content
- prep_formatting
- prep_formatting2
- prep_hot_encoding_prep - upto Re-hot encoding 
- prep_adding_and_deleting_columns
- prep_final

To adjusting thresholds or to remove extra columns later:
hot_encoding_prep - from Re-hot encoding 
adding_and_deleting_columns


### Modelling: 
For more detailed information see: README_modelling.md

prepare for models:
- prep_rebalancing

Running models - here order in not as important 

- pred_random_forest
- pred_crossval_randomforest
- pred_neural_networks
- pred_nlp
- pred_xgboost_tries
- pred_voter
