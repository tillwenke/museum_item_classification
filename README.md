# Classification of items in Estonian Museums
Part of this [kaggle challenge](https://www.kaggle.com/competitions/caps-in-museums-data-part-2)

In [Column Discovery](https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing) our findings and resulted feature engineering tasks per feature are captured.

# Getting started
Embed the cloned project in a [PyVenv](https://docs.python.org/3/library/venv.html) (or conda env if you prefer) and install the requiremnts listed in requirements.txt

Due to large files you have to get trained models and text embeddings from [here](https://drive.google.com/drive/folders/1ZKOynMoLZd0R_0Q8kB-AwELzFwo50atb?usp=sharing) and paste them respectively in models/ and data/text_embeddings

For getting a quick overview check out the respective folder.

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

# ReadMe for Data Prep:

All notebooks can be run from start to finish.

## prep_combining_test_and_train

This notebook combines the train and test dataset provided by Kaggle into one collective dataset to 
used for the data prepartion

## prep_translation

Text data is translated in this notebook.

## prep_hot_encoding

This notebook is used to hot encode our text based categorical features.
This notebook also includes data minipulation needed before the hot encoding is applied:
- For some features such as colour, entries are grouped into larger more useful groups
- Some features have been thresholded so that we can limit the number of additional hot encoded columns:

The group_under_threshold(data, col, threshold) function dose the following:
- goes through the value_counts and selects all the values below the threshold
- labels these as 'below_tres' and groups these together
So that during the hot_encoding we only have columns for values that appear at least as often as the threshold

This notebook also includes a section for adjusting the thresholds at a later date

## prep_mapping

## prep_units_sizes

For feature-engineering/ reassembling the parameter-unit-size features.

## prep_split_feature_content

Some features contained lists of values or multiple information in some other way - those are split up into separate features here.

## prep_whitespace_deletion

Some cells conatained whitespaces instead of '' or NaN.

## prep_xgboost

Rename columns for xgboost (cant deal with <>[] in feature names).

## prep_formatting
'Start' and 'End' have multiple different formats. The goal is to put the information into a uniform and useable format.
This is done by splitting both columns up in 3 columns each. For days and months there is an indicator now that shows whether these values are given. The year is in an own column and it is 0 if there as none given. The original columns were deleted from the data.

## prep_formatting2
in this notebook, brackets are removed from the column "event_type". Also, a regular expression found in a value was removed.

## prep_adding_and_deleting_columns:

This notebook is used to remove certain columns from the prep datset and add additional columns for whether a 
feature has a value(1) or an NaN entery(0).

## prep_final

Features that are not needed are deleted and NaNs from different columns are changed and finally splits into test and train data for predicting phase.

## prep_rebalancing

In this notebook, rebalancing takes place with ROS and SMOTE on the whole dataset. The rebalanced datasets are saved as .csv-files. 

# ReadMe for Modelling:

## crossval_randomforest:

This notebook uses cross validation to train randomforest models, including through Gridsearch

## pred_xgboost

In this notebook, a model is trained using xgboost. At the start, the variable rebalanced is set to false and should be set as true if rebalancing via SMOTE should be applied to the train-data. Also includes the code to run a wandb-sweep.

## pred_random_forest

In this notebook, a model is trained using a random forest classifier. At the start, the variable rebalanced is set to false and should be set as true if rebalancing via SMOTE should be applied to the train-data. Also includes the code to run a wandb-sweep.

## pred_neural_network

This notebook is used to train a neural network on tabular training data.

## prep_nlp

This notebook is used to train different models using GPT-3 text embeddings from textual data. It also provides the code to gain GPT-3 embedding vectors. Due to restrictions in OpenAI's free trial this has been an overnight-task (babbage/ curie embeddings). In December 2022 gaining the babbage and curie (models on it performed slightly better) embeddings once each ate up my 18$ free trial amount.

## pred_voter

In this notebook the final model assembly takes place and different models are combined to vote (hard, soft) on classification and some text-keyword -> type classifiers are tried.

