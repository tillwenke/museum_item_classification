from setup_general import *
from prep_helpers import *

def get_data(feat_percent_cut, feat_freq_cut):
    
    data = combined_intermediate_ready.copy()

    # best found combination (local optimum on 500 estimators)
    perc = feat_percent_cut/100
    threshold_sum = len(data) * perc
    min_freq = feat_freq_cut

    tech = helpers.col_collection(data, 'technique_')
    mat = helpers.col_collection(data, 'material_')
    size = data.columns[data.columns.str.contains('IN')]

    features = [tech,mat,size]

    for feat in features:
        frequencies = {}
        for col in feat:
            frequencies[col] = data[col].sum()
        frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
        instance_sum = 0
        for col in frequencies:
            frequency = frequencies[col]
            #if instance_sum > threshold_sum or frequency < min_freq:
            if frequency < min_freq:
                data.drop(columns=[col], inplace=True)
            instance_sum += frequency

            
    ## hot encoding & thresholding
    # categorical columns
    # already encoded
    # material, technique, unit, size, value

    cols = ['musealia_additional_nr', 'collection_mark', 'musealia_mark', 'museum_abbr', 'before_Christ', 'is_original', 'class', 'parish', 'state',  'event_type', 'participants_role', 'parish', 'color', 'collection_additional_nr', 'damages', 'participant', 'location', 'name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text', 'country', 'city_municipality']

    text_features = ['name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text']
    for col in cols:
        data[col] = data[col].fillna('nan')
        instance_sum = 0
        val_counts = data[col].value_counts()
        values_to_group = []
        for idx, name in enumerate(val_counts.index):
            frequency = val_counts[idx]
            if instance_sum > threshold_sum or frequency < min_freq:
                values_to_group.append(name)

            instance_sum += frequency
        data[col] = data[col].apply(lambda x: 'uncommon' if (x in values_to_group) else x)

    # one hot encoding
    data = pd.get_dummies(data, columns=cols)
        
    ## Delete unneeded features


    data.drop(columns=['full_nr','country_and_unit','parameter','unit','value'], inplace=True)

    ## continous numeric features (nan -> 0)
    data = data.replace(np.nan, 0)
    ## rename for xgboost (cant deal with <>[] in feature names)
    for i in data.columns:
        if '>' in i:
            data.rename(columns={i:i.replace('>','')}, inplace=True)
        if '<' in i:
            data.rename(columns={i:i.replace('<','')}, inplace=True)
        if ']' in i:
            data.rename(columns={i:i.replace(']','')}, inplace=True)
        if '[' in i:
            data.rename(columns={i:i.replace('[','')}, inplace=True)

    ## resplit test/train
    train = data.loc[data['source']=='train'].drop('source',axis=1)

    # modify types
    train['type'] = train['type'].replace('fotonegatiiv, fotonegatiiv', 'fotonegatiiv')    

    # resplit test/train
    train, val = train_test_split(train, test_size=0.3, random_state=0)
    test = data.loc[data['source']=='test'].drop('source',axis=1)

    return train, val, test

#function to have resamplers resample to specific number of samples per class
def by_num(y, min_samples):
    b = Counter(y).values()
    a = Counter(y).keys()
    a = list(a)
    b = list(b)

    if min_samples > max(b):
        min_samples = max(b)

    for i in range(len(a)):
        if b[i] < min_samples :
            b[i] = min_samples
    return dict(zip(a, b))

#function to have resamplers resample to specific number of samples per class
def by_perc(y, increase_perc):
    a = Counter(y).keys()
    b = Counter(y).values()
    a = list(a)
    b = list(b)

    max_samples = max(b)

    for i in range(len(b)):
        new_samples = int(b[i] * (1 + increase_perc/100))
        if new_samples > max_samples:
            b[i] = max_samples
        else:
            b[i] = new_samples
    return dict(zip(a, b))


def rebalancing(X, y, reb_method, strategy, by_value):

    if strategy == 'perc':
        sampling_strategy = by_perc
    else:
        sampling_strategy = by_num
    
    if reb_method == 'smote':
        balancer = SMOTE(sampling_strategy=sampling_strategy(y,by_value), random_state=0)
    elif reb_method == 'ros':
        balancer = RandomOverSampler(sampling_strategy=sampling_strategy(y,by_value), random_state=0)
    else:
        return X, y

    X_res, y_res = balancer.fit_resample(X, y)

    return X_res, y_res


project = 'rf'

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_f1_macro'},
    'parameters': 
    {
        'min_samples_split': {'values': [2, 5, 7, 10, 20, 40, 100, 200]},
        'max_depth': {'values': [3, 6, 10, 25, 50, 75, 100, 150, 200, 500, 1000, None]},
        'min_samples_leaf': {'values': [1, 2, 6, 10, 20, 40, 50, 70, 100, 200, 500, 1000]},
        'n_estimators': {'values': [100, 200, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000]},
        'criterion': {'values': ['gini', 'entropy', 'log_loss']},
        'max_features': {'values': [None, 'sqrt', 'log2']},
        'feat_percent_cut': {'min': 50, 'max': 100},
        'feat_freq_cut': {'min': 1, 'max': 15},
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

    # note that we define values from `wandb.config` instead 
    # of defining hard values 
    min_samples_split = wandb.config.min_samples_split
    max_depth = wandb.config.max_depth
    min_samples_leaf = wandb.config.min_samples_leaf
    n_estimators = wandb.config.n_estimators
    max_features = wandb.config.max_features
    criterion = wandb.config.criterion
    feat_percent_cut = wandb.config.feat_percent_cut
    feat_freq_cut = wandb.config.feat_freq_cut
    reb_method = wandb.config.reb_method
    rebalance = wandb.config.rebalance
    class_weight = wandb.config.class_weight

    # -------------------------- data prep code  -------------------------------------

    print('data prep')
    train, val, test = get_data(feat_percent_cut=feat_percent_cut, feat_freq_cut=feat_freq_cut)

    print('balancing')
    print(rebalance)
    strategy, by_value = rebalance
    print(strategy, by_value)

    #val = val_est_prepared.copy()

    X_train = train.drop('type', axis=1)
    y_train = train.type

    """"
    X_val = val.drop('type', axis=1)
    y_val = val.type
    """

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_train)

    y_train = label_encoder.transform(y_train)
    #y_val = label_encoder.transform(y_val)
    

    # -------------------------- usual training code starts here  -------------------------------------
    print('training')
    
    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,\
         max_features=max_features, min_samples_split=min_samples_split, class_weight=class_weight, random_state=0, n_jobs=-1)

    skf = StratifiedKFold(n_splits=4)

    val_acc = []
    val_f1_macro = []

    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print('fold', i)
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # replace uncommon types
        unique, counts = np.unique(y_train_fold, return_counts=True)
        # 6 to have 5 samples per class left for standard knn in smote
        # -> uncommon classes become 100
        for i in np.argwhere(counts < 6):
            y_train_fold[y_train_fold == i[0]] = 100

        X_train_fold, y_train_fold = rebalancing(X_train_fold, y_train_fold, reb_method=reb_method, strategy=strategy, by_value=by_value)

        rfc.fit(X_train_fold, y_train_fold)

        y_pred = rfc.predict(X_test_fold)
        val_acc.append(accuracy_score(y_test_fold, y_pred))
        val_f1_macro.append(f1_score(y_test_fold, y_pred, average='macro'))

    crossval_acc = np.mean(val_acc)
    crossval_f1_macro = np.mean(val_f1_macro)

    # -------------------------- ends here  -------------------------------------
    

    wandb.log({
      'val_acc': crossval_acc,
      'val_f1_macro': crossval_f1_macro
    })

# Start sweep job.
wandb.agent(sweep_id, function=main, count=1000)