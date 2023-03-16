from setup_general import *
from setup_embedding import *
from prep_helpers import *

#from sklearnex import patch_sklearn
#patch_sklearn()

def get_data(feat_percent_cut=1.0, feat_freq_cut=0):
    # adapted from preparation.ipynb -> everything after "rescaling"

    # rather cut from both ends
    
    data = combined_intermediate_ready.copy()

    perc = feat_percent_cut/100
    threshold_sum = len(data) * perc
    min_freq = feat_freq_cut

    tech = col_collection(data, 'technique_')
    mat = col_collection(data, 'material_')
    size = data.columns[data.columns.str.contains('IN')]

    features = [tech,mat,size]
    
    # !!! should not be allowed bc includes knowledge about test set
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

    cols = ['musealia_additional_nr', 'collection_mark', 'musealia_mark', 'museum_abbr', 'before_Christ', 'is_original', 'class', 'state', 'event_type', 'participants_role', 'parish', 'color', 'collection_additional_nr', 'damages', 'participant', 'location', 'name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text', 'country', 'city_municipality']

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

def get_curie():
    #splitting
    data = combined_intermediate_ready.copy()
    trainval = data.loc[data['source']=='train']
    test = data.loc[data['source']=='test']
    train, val = train_test_split(trainval, test_size=0.3, random_state=0)

    trainval_curie = curie[curie.source == 'train'].drop(columns=['source'])
    test_curie = curie[curie.source == 'test'].drop(columns=['source'])

    train_curie = pd.DataFrame.join(train[['element_count']], trainval_curie)
    train_curie.dropna(axis=0, inplace=True)
    train_curie.drop(columns=['element_count'], inplace=True)
    print(len(train_curie))

    val_curie = pd.DataFrame.join(val[['element_count']], trainval_curie)
    val_curie.dropna(axis=0, inplace=True)
    val_curie.drop(columns=['element_count'], inplace=True)
    print(len(val_curie))

    test_curie = pd.DataFrame.join(test[['element_count']], test_curie)
    test_curie = test_curie.drop(columns=['type'])
    test_curie.dropna(axis=0, inplace=True)
    test_curie.drop(columns=['element_count'], inplace=True)
    print(len(test_curie))

    return train_curie, val_curie, test_curie

def get_bow(max_n_gram=2, max_features=1000):
    dataset = text_est.copy()
    stop_words = stopwords_est

    CountVec = TfidfVectorizer(ngram_range=(1,max_n_gram), stop_words=stop_words, max_features=max_features)
    # to use bigrams ngram_range=(2,2)
    Count_data = CountVec.fit_transform(dataset.text_features)
    #create dataframe
    bow=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())

    bow = bow.add_prefix('word_')
    bow.index = dataset.index
    bow = bow.join(dataset[['source', 'type']])

    #splitting
    data = combined_intermediate_ready.copy()
    trainval = data.loc[data['source']=='train']
    test = data.loc[data['source']=='test']
    train, val = train_test_split(trainval, test_size=0.3, random_state=0)

    trainval_bow = bow[bow.source == 'train'].drop(columns=['source'])
    test_bow = bow[bow.source == 'test'].drop(columns=['source'])

    train_bow = pd.DataFrame.join(train[['element_count']], trainval_bow)
    train_bow.dropna(axis=0, inplace=True)
    train_bow.drop(columns=['element_count'], inplace=True)
    print(len(train_bow))
    
    val_bow = pd.DataFrame.join(val[['element_count']], trainval_bow)
    val_bow.dropna(axis=0, inplace=True)
    val_bow.drop(columns=['element_count'], inplace=True)
    print(len(val_bow))

    test_bow = pd.DataFrame.join(test[['element_count']], test_bow)
    test_bow = test_bow.drop(columns=['type'])
    test_bow.dropna(axis=0, inplace=True)
    test_bow.drop(columns=['element_count'], inplace=True)
    print(len(test_bow))

    return train_bow, val_bow, test_bow

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


def training(train, clf, reb_method, rebalance):
    strategy, by_value = rebalance    

    X_train = train.drop('type', axis=1)
    y_train = train.type.copy(deep=True)

    # replace uncommon types -> gives small improvement in acc & f1 (investigate further?)
    # 8 to have 5 samples per class left for standard knn in smote after 4 fold cv
    for type in y_train.value_counts()[y_train.value_counts() < 8].index:
        y_train[y_train == type] = 'uncommon_type'
 
    label_encoder = LabelEncoder().fit(y_train)

    y_train = label_encoder.transform(y_train)

    # -------------------------- usual training code starts here  -------------------------------------
    print('training')

    skf = StratifiedKFold(n_splits=4)

    val_acc = []
    val_f1_macro = []

    start_time = time.time()

    time_reb = 0
    time_train = 0
 
    for k, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]        

        start_time = time.time()
        X_train_fold, y_train_fold = rebalancing(X_train_fold, y_train_fold, reb_method=reb_method, strategy=strategy, by_value=by_value)
        time_reb += time.time() - start_time

        print('fold', k)
        start_time = time.time()
        clf.fit(X_train_fold, y_train_fold)
        time_train += time.time() - start_time

        y_pred = clf.predict(X_test_fold)
        val_acc.append(accuracy_score(y_test_fold, y_pred))
        val_f1_macro.append(f1_score(y_test_fold, y_pred, average='macro'))


    crossval_acc = np.mean(val_acc)
    crossval_f1_macro = np.mean(val_f1_macro)

    monitoring = dict()
    monitoring['crossval_acc'] = crossval_acc
    monitoring['crossval_f1_macro'] = crossval_f1_macro
    monitoring['time_reb'] = time_reb
    monitoring['time_train'] = time_train

    return monitoring