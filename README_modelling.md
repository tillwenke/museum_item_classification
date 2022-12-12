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
