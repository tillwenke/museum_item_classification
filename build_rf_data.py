from sweep_utilities import *

name = "blue"

#data specific
feat_percent_cut = wandb.config.feat_percent_cut
feat_freq_cut = wandb.config.feat_freq_cut

# rebalancing specific
reb_method = wandb.config.reb_method
rebalance = wandb.config.rebalance

# model specific
class_weight = wandb.config.class_weight
min_samples_split = wandb.config.min_samples_split
max_depth = wandb.config.max_depth
min_samples_leaf = wandb.config.min_samples_leaf
n_estimators = wandb.config.n_estimators
max_features = wandb.config.max_features
criterion = wandb.config.criterion

rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth, min_samples_leaf=min_samples_leaf,\
    max_features=max_features, criterion=criterion,class_weight=class_weight, random_state=0, n_jobs=-1)

print('data prep')
train, val, test = get_data(feat_percent_cut=feat_percent_cut, feat_freq_cut=feat_freq_cut)

monitoring = training(train, rfc, reb_method, rebalance)

print(monitoring['crossval_acc'], monitoring['crossval_f1_macro'], monitoring['time_reb'], monitoring['time_train'])

pickle.dump(rfc, open(f'./models/rf/{name}', 'wb'))


