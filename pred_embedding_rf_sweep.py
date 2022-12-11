from setup_general import *
from setup_embedding import *

print('loaded data')

df = curie_train.copy()
 
X_train, X_test, y_train, y_test = train_test_split(
    list(df.curie_similarity.values),
    df.type,
    test_size = 0.3,
    random_state=0
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'split': {'values': [2, 5, 7, 10, 12, 15, 20]},
        'depth': {'values': [3, 6, 10, 50, 100, 500, 1000, 2000]},
        'leaf': {'values': [2]},
        'estimators': {'values': [100, 200, 350, 500, 1000, 1500, 2000]},
        'features': {'values': [None]},

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
    

    # -------------------------- usual training code starts here  -------------------------------------
    
    clf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, min_samples_leaf=leaf, max_features=features, min_samples_split=split, random_state=42)
    print('run')
    clf.fit(X_train, y_train)
    print('ran')
    preds = clf.predict(X_test)

    val_acc = accuracy_score(y_test, preds)
    
    y_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)

    #report = classification_report(y_test, preds)
    #print(report)

    #clf.save_model('models/curie_model.json')

    # -------------------------- ends here  -------------------------------------
    

    wandb.log({
      'train_acc': train_acc,
      'val_acc': val_acc,
    })

# Start sweep job.
wandb.agent(sweep_id, function=main)
 