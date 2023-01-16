# Classification of items in Estonian Museums
Part of this [kaggle challenge](https://www.kaggle.com/competitions/caps-in-museums-data-part-2); [original data](https://opendata.muis.ee/)

Baseline: 100 tree RF: 30 % validation set [accuracy on kaggle test set: private: 0.901 public: 0.90266]
                               precision    recall  f1-score   support

                      ajakiri       1.00      0.71      0.83        31
                      ajaleht       0.42      0.35      0.38        23
                        album       0.25      0.14      0.18         7
          arheoloogiline leid       0.98      1.00      0.99       259
             aukiri/auaadress       1.00      0.38      0.55         8
                  diapositiiv       1.00      0.94      0.97        17
           digitaalne kujutis       0.98      1.00      0.99        44
                     dokument       0.58      0.70      0.63       125
                          ehe       1.00      0.50      0.67         6
                         foto       0.94      0.98      0.96      1114
              foto, postkaart       1.00      0.24      0.38        17
                 fotomaterjal       1.00      0.73      0.84        11
                 fotonegatiiv       0.98      1.00      0.99       664
   fotonegatiiv, fotonegatiiv       0.00      0.00      0.00         8
                     graafika       0.83      0.97      0.89        69
                helisalvestis       1.00      1.00      1.00        17
                    joonistus       1.00      0.53      0.69        17
                     kalender       1.00      0.81      0.90        16
                   karikatuur       1.00      0.71      0.83        14
                kaustik/vihik       0.00      0.00      0.00         1
                         kava       0.78      0.84      0.81       232
         kavand/joonis/eskiis       0.98      0.99      0.99       234
                         kiri       0.62      0.75      0.68       198
              kiri, postkaart       0.50      0.67      0.57         3
                        kleit       0.45      1.00      0.62         5
                         kott       1.00      0.25      0.40         4
                        kutse       1.00      0.29      0.44        35
                     käsikiri       0.84      0.76      0.80       144
käsikiri, laul/ vokaalmuusika       0.79      0.95      0.86        58
        käsikiri, muusikateos       0.40      0.10      0.16        20
             käsikiri, noodid       1.00      1.00      1.00         2
                  laegas/karp       0.50      0.50      0.50         6
                   lina/linik       1.00      0.83      0.91         6
                    lint/pael       0.00      0.00      0.00         7
                         maal       0.83      0.71      0.77         7
                        medal       1.00      0.90      0.95        31
            muusikainstrument       1.00      0.91      0.95        11
                      märkmed       0.00      0.00      0.00         5
                         münt       0.98      1.00      0.99        56
                       noodid       0.84      0.93      0.88        71
                         nukk       0.00      0.00      0.00         1
                     nõu/anum       1.00      0.80      0.89        10
                        paber       0.57      0.50      0.53         8
                       pakend       1.00      0.25      0.40         4
                       pitsat       1.00      0.75      0.86         4
         pitser/templijäljend       0.99      1.00      0.99        66
                       plakat       0.99      1.00      0.99       221
                    postkaart       0.50      0.20      0.28        51
                       raamat       0.94      0.97      0.95       154
                    silt/märk       0.47      0.75      0.58        12
                    skulptuur       0.00      0.00      0.00         3
                    telegramm       0.50      0.45      0.48        22
                    trükinoot       1.00      0.29      0.44         7
                    tunnistus       0.00      0.00      0.00         1
                  väiketrükis       0.83      0.61      0.70        33

                     accuracy                           0.89      4200
                    macro avg       0.73      0.61      0.64      4200
                 weighted avg       0.89      0.89      0.89      4200

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

