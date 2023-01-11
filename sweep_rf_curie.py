from sweep_utilities import *

project = 'rf_curie'
# 0. narrowing down from all params at once too complex
# 1. use 1000 estimators to find right dataset
# 2. tune rf parameters

# Define sweep config
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
                        ('num',2500),('num',3000)]},
        'class_weight': {'values': [None, 'balanced']}
     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

def main():
    run = wandb.init(project=project)

    # rebalancing specific
    reb_method = wandb.config.reb_method
    rebalance = wandb.config.rebalance

    # model specific
    class_weight = wandb.config.class_weight

    rfc = RandomForestClassifier(n_estimators=1000, class_weight=class_weight, random_state=0, n_jobs=-1)

    print('data prep')
    train, val, test = get_curie()

    monitoring = training(train, rfc, reb_method, rebalance)
    
    wandb.log({
      'val_acc': monitoring['crossval_acc'],
      'val_f1_macro': monitoring['crossval_f1_macro'],
      'rebalancing_time': monitoring['time_reb'],
      'training_time': monitoring['time_train']
    })

# Start sweep job.
wandb.agent(sweep_id, function=main, count=10000)