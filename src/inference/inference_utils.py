from operator import add
import numpy as np
import pandas as pd
import re
from numpy import nan
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def text_to_bow(path, max_n_gram=2, max_features=2000):
    dataset = pd.read_csv(path, index_col='id')

    with open('data/general/estonian-stopwords.txt') as file:
        lines = [line.rstrip() for line in file]
    stopwords_est = lines
    stop_words = stopwords_est

    CountVec = TfidfVectorizer(ngram_range=(1,max_n_gram), stop_words=stop_words, max_features=max_features)
    # to use bigrams ngram_range=(2,2)
    Count_data = CountVec.fit_transform(dataset.text_features)
    #create dataframe
    bow=pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names_out())

    bow = bow.add_prefix('word_')
    bow.index = dataset.index
    return bow

def vote(preds):
    # only if there is a safe prediction
    if preds[-1][0] == -1:
        preds = preds[:-1]
        
    res = np.sum(preds, axis=0)
    return np.argmax(res), np.max(res)

# KASPARS WORK

# using for our preparation?
"""
# convert to numeric
def replace_value(value: str):
    if pd.isnull(value):
        return value
    return np.float64(value.replace(',', '.'))


# convert to numeric and only keep year part
def replace_start_end(value: str):
    if pd.isnull(value):
        return value
    if re.match('^d?ddd$', value):
        return int(value)
    if re.match('dddd$', value):
        return int(value[-4:])
    elif not value[0].isdigit():
        return int(f'19{value[-2:]}')
    else:
        return nan


def extract_year_from_name(row):
    name = row['name']
    start = row['start']
    if pd.isnull(start) and not pd.isnull(name):
        match = re.search('\d\d\d\d', name)
        if match:
            start = match.group()
    return start
"""

# not needed
"""
def preprocess_dataframe(df, submission=False):
    categorical_cols = ['material', 'location', 'before_Christ', 'country_and_unit', 'technique', 'parameter',
                        'museum_abbr', 'damages', 'state', 'color', 'event_type', 'collection_mark']
    categorical_cols += ['unit', 'participants_role', 'participant', 'musealia_mark']

    # just keeping track what values are used
    numeric_cols = ['start', 'end', 'value', 'collection_queue_nr', 'is_original', 'ks', 'element_count',
                    'musealia_seria_nr', 'musealia_queue_nr']

    dropped_cols = ['id', 'parish']  # can't use
    dropped_cols += ['full_nr', 'class', 'collection_additional_nr', 'additional_text', 'text', 'initial_info',
                     'musealia_additional_nr']  # 'commentary','name', 'legend'

    if not submission: dropped_cols.append('type')

    df['start'] = df['start'].apply(replace_start_end)
    df['end'] = df['end'].apply(replace_start_end)
    df['value'] = df['value'].apply(replace_value)
    df['start'] = df[['name', 'start']].apply(extract_year_from_name, axis=1)

    df = df.drop(columns=dropped_cols)
    df = pd.get_dummies(df, columns=categorical_cols)
    df = df.fillna(0)
    return df
"""

def extract_label_from_comment(row):
    # comment #################################################
    comment = row['commentary']

    if not pd.isnull(comment):
        comment = str(comment).lower()

        comment_dict = {
            'lakk': 'pitser/templijäljend',
            'must-valge negatiiv': 'fotonegatiiv',
            'pitserilakk': 'pitser/templijäljend',
            'käepide': 'pitsat',
            'перф': 'fotonegatiiv',
            'fotoemulsioon': 'fotomaterjal',
            'plakat':'plakat'
        }
        for key, val in comment_dict.items():
            if comment.startswith(key):
                return val

        if re.match('^\d,\d\d\sg$', comment):
            return 'münt'

        if 'diapositiiv' in comment:
            return 'diapositiiv'

    # name #################################################
    name = row['name']

    if not pd.isnull(name):
        name = str(name).lower()
        if name == ['denaar', 'killing', 'penn', 'schilling', '1/2 örtug', 'dirhem', 'fyrk']:
            return 'münt'

        for val in ['medal', 'plakat', 'märkmed', 'maal', 'kiri', 'kleit', 'kava', 'joonistus', 'graafika', 'dokument',
                    'ajakiri', 'telegramm', 'skulptuur', 'raamat', 'postkaart', 'nukk', 'skulptuur', 'käsikiri']:
            if name.startswith(val):
                return val

        name_dict = {
            'kaustik': 'kaustik/vihik',
            'vihik': 'kaustik/vihik',
            'reprofoto': 'diapositiiv',
        }
        for key, val in name_dict.items():
            if name.startswith(key):
                return val
    return nan

def replace_predictions(labels, pred):
    result = np.array(pred, copy=True)
    for i, label in enumerate(labels):
        if not pd.isnull(label) and label != 0:
            result[i] = label
    return result