# ReadMe for Data Prep:

## combining_test_and_train:
This notebook combines the train and test dataset provided by Kaggle into one collective dataset to 
used for the data prepartion

## hot_encoding_prep:

This notebook is used to hot encode our text based categorical features.
This notebook also includes data minipulation needed before the hot encoding is applied:
- For some features such as colour, entries are grouped into larger more useful groups
- Some features have been thresholded so that we can limit the number of additional hot encoded columns:

The group_under_threshold(data, col, threshold) function dose the following:
- goes through the value_counts and selects all the values below the threshold
- labels these as 'below_tres' and groups these together
So that during the hot_encoding we only have columns for values that appear at least as often as the threshold

This notebook also includes a section for adjusting the thresholds at a later date

## adding_and_deleting_columns:

This notebook is used to remove certain columns from the prep datset and add additional columns for whether a 
feature has a value(1) or an NaN entery(0).

