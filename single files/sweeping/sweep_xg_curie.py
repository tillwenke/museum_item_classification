from train_utils import *


project = 'xg_curie'

# Define sweep config
# from https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
# https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
# step by step https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_f1_macro'},
    'parameters': 
    {
        'reb_method': {'values': ['none', 'smote', 'ros']},
        'rebalance': {'values': [('perc',10),('perc',20),('perc',30),('perc',40),('perc',50),('perc',60),('perc',70),('perc',80),\
            ('perc',90),('perc',100),('perc',200),('perc',300),('perc',400),('perc',500),('perc',600),('perc',700),('perc',800),\
                ('perc',900),('perc',1000),('perc',2000),('perc',5000),('perc',10000),('perc',50000), ('num',10),('num',20),('num',50),\
                    ('num',70),('num',100),('num',200),('num',300),('num',400),('num',500),('num',700),('num',1000),('num',1500),('num',2000),\
                        ('num',2500),('num',3000)]}
     }
}

"""
'max_depth': {'min': 3, 'max': 1000},
'gamma': {'min': 0.0, 'max': 9.0},
'learning_rate': {'min': 0.0, 'max': 1.0},
'reg_alpha': {'min': 0, 'max': 180},
'reg_lambda' : {'min': 0, 'max': 100},
'colsample_bytree' : {'min': 0.1, 'max': 1.0},
'subsample' : {'min': 0.1, 'max': 1.0},
'min_child_weight' : {'min': 0, 'max': 100},       
'n_estimators': {'values': [100, 200, 500, 800, 1000, 1500, 2000, 3000, 5000]},
"""

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

def main():
    run = wandb.init(project=project)

    #rebalancing specific
    reb_method = wandb.config.reb_method
    rebalance = wandb.config.rebalance

    #model specific
    """
    max_depth = wandb.config.max_depth
    gamma = wandb.config.gamma
    learning_rate = wandb.config.learning_rate
    reg_alpha = wandb.config.reg_alpha
    reg_lambda = wandb.config.reg_lambda
    colsample_bytree = wandb.config.colsample_bytree
    subsample = wandb.config.subsample
    min_child_weight = wandb.config.min_child_weight
    n_estimators = wandb.config.n_estimators

    xg = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda,\
        colsample_bytree=colsample_bytree, subsample=subsample, tree_method='gpu_hist', random_state=0, n_jobs=-1)
    """
    xg = XGBClassifier(tree_method='gpu_hist', random_state=0, n_jobs=-1)

    print('data prep')
    train, val, test = get_curie()

    monitoring = training(train, xg, reb_method, rebalance)    

    wandb.log({
      'val_acc': monitoring['crossval_acc'],
      'val_f1_macro': monitoring['crossval_f1_macro'],
      'rebalancing_time': monitoring['time_reb'],
      'training_time': monitoring['time_train']
    })

# Start sweep job.
wandb.agent(sweep_id, function=main, count=10000)