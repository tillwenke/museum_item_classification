# ReadMe for Modelling:

## crossval_randomforest:

This notebook uses cross validation to train randomforest models, including through Gridsearch

## pred_xgboost

In this notebook, a model is trained using xgboost. At the start, the variable rebalanced is set to false and should be set as true if rebalancing via SMOTE should be applied to the train-data. Also includes the code to run a wandb-sweep.

## pred_random_forest

In this notebook, a model is trained using a random forest classifier. At the start, the variable rebalanced is set to false and should be set as true if rebalancing via SMOTE should be applied to the train-data. Also includes the code to run a wandb-sweep.

## pred_neural_network

This notebook is used to train a neural network on train data.

## prep_nlp

This notebook is used to train different models using GPT-3 text embeddings from textual data.

## pred_voter

In this notebook the different models are combined to vote on classification and some text-keyword -> type classifiers are tried.
