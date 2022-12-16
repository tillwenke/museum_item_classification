from setup_general import *
from setup_embedding import *

print('loaded data')

df = train_bal_curie.copy()
 
features = df.drop('type', axis=1)
labels = df.type
# at least xgboost cannot deal with string labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
labels = label_encoder.transform(labels)
X_train = features
y_train = labels

# Define sweep config
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'split': {'values': [2, 3, 4, 5, 7, 10, 12, 15, 20]},
        'depth': {'values': [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 500, None]},
        'leaf': {'values': [1, 2, 4]},
        'estimators': {'values': [100, 200, 350, 500, 700, 1000, 1500, 2000]},
        'features': {'values': [None, 'sqrt', 'log2']},
        'criterion':{'values':['gini', 'entropy']},

     }
}

# Initialize sweep by passing in config. (Optional) Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project='nlp')

def main():
    run = wandb.init(project="nlp")

    # note that we define values from `wandb.config` instead 
    # of defining hard values 
    split = wandb.config.split
    depth = wandb.config.depth
    leaf = wandb.config.leaf
    estimators = wandb.config.estimators
    features = wandb.config.features
    criterion = wandb.config.criterion
    

    # -------------------------- usual training code starts here  -------------------------------------
    
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, criterion=criterion, min_samples_leaf=leaf, max_features=features, min_samples_split=split, random_state=42)
    print('run')
    cross_val_acc = np.mean(cross_val_score(clf, X_train, y_train, cv=5))
    print('ran')
    # -------------------------- ends here  -------------------------------------    

    wandb.log({
      'val_acc': cross_val_acc,
    })

# Start sweep job.
wandb.agent(sweep_id, function=main)
 
