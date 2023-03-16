from setup_general import *

for counter in range(0,11):
    lang = 'est'
    if lang == 'en':
        data = combined_data_fully_translated.copy()
    if lang == 'est':
        data = combined_data.copy()

    numeric_features = ['ks', 'musealia_seria_nr', 'musealia_queue_nr', 'collection_queue_nr', 'element_count']
    cols = ['musealia_additional_nr', 'collection_mark', 'musealia_mark', 'museum_abbr', 'before_Christ', 'is_original', 'class', 'parish', 'state',  'event_type', 'participants_role', 'parish', 'color', 'collection_additional_nr', 'damages', 'participant', 'location']
    text_features = ['name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text']

    ## country_and_unit - technique (whitespace deletion)
    def empty_to_nan(item):
        if type(item) == str:
            item = item.strip()
            if item == '':
                return np.nan
            else:
                return item
        else:
            return item

    data['country_and_unit'] = data.apply(lambda x: empty_to_nan(x['country_and_unit']), axis=1)
    data['technique'] = data['technique'].apply(lambda x: x.strip() if (type(x) == str) else x)

    ## event_type (brackets)
    def strip_brackets(x):
        x = str(x)
        x = x.strip('< >')
        if '\u200b' in x:
            x = x.replace('\u200b', '')
        return x

    data['event_type'] = data['event_type'].apply(strip_brackets)

    # Feature specific engineering
    ## units - sizes -values
    # Finish unit translation/ unification &  values to float
    data['value'] = data['value'].apply(lambda x: float(x.replace(',', '.')) if type(x) == str else x)

    if counter == 1:
        # unify units
        data['unit'] = data['unit'].replace('10 x 15 cm','100 x 150 mm')

        # mm to cm
        data['value'] = data.apply(lambda item: item['value'] / 10 if item['unit'] == 'mm' else item['value'], axis=1)
        data['unit'] = data['unit'].replace('mm','cm')
        data['value'] = pd.to_numeric(data['value'])

        # Combine parameter, unit & w/h values to value
        def get_squared(item):
            if ' x ' in item:
                return item + '²'
            else:
                return item

        def extract_width_height_from_unit_to_value(item):
            unit = item[0]
            value = item[1]
            if ' x ' in unit:
                split = unit.split(' ')
                x = split[0]
                y = split[2]
                real_unit = split[3]
                real_value = [x,y]        
                return [real_unit, real_value]

            else:
                return [unit, value]

        data['unit'] = data['unit'].replace(np.nan,'*')
        data['parameter'] = data['parameter'].replace(np.nan,'*')
        data['unit'] = data['unit'].apply(lambda x: get_squared(x))
        # execution order is important
        data['value'] = data.apply(lambda item: extract_width_height_from_unit_to_value(item[['unit','value']])[1], axis=1)
        data['unit'] = data.apply(lambda item: extract_width_height_from_unit_to_value(item[['unit','value']])[0], axis=1)
        data['parameter_and_unit'] = data['parameter'] + ' IN ' + data['unit']

        # parameter_and_units as single features with respective values
        # parameter_and_unit turned into one hot encoded features
        data= pd.get_dummies(data, columns=['parameter_and_unit'], prefix='', prefix_sep='')

        def extract_value(value, present):
            if present == 1:
                return value
            else:
                return 0

        #  for all new "parameter with unit" columns put the value in the column where a 1 is - others are 0 and remain 0
        for column in data.columns:
            if ' IN ' in column and '*' not in column:
                data[column] = data.apply(lambda item: extract_value(item['value'], item[column]), axis=1)

        # make all size features numeric
        def extract_height_width(item):    
            if item == 0:
                return [0.0,0.0]
            else:        
                return [float(i) for i in item]
                

        for column in data.columns:
            # all the parameter with unit columns that contain arrays that are represeted as strings
            if (' IN ' in column) and (data[column].dtype == object):
                data[column + '_height'] = data.apply(lambda item: extract_height_width(item[column])[0], axis=1)
                data[column + '_width'] = data.apply(lambda item: extract_height_width(item[column])[1], axis=1)
                pd.to_numeric(data[column + '_height'])
                pd.to_numeric(data[column + '_width'])
                data = data.drop(column, axis=1)

        for column in data.columns:
            if (' IN ' in column):
                data[column] = data[column].replace(np.nan,0)

        data.drop(columns=['parameter','unit','value'], inplace=True)
    else:
        cols.extend(['parameter','unit'])
        numeric_features.append('value')


    ## country_unit - material - technique - location (splitting for features including multiple information)
    if counter == 2:
        # country_unit
        def extract_city_country(item):      
            if (type(item) == str):
                item = item.strip()
                # there are some empty (non-nan) values
                if (item == ''):
                    return [float('nan'), float('nan')]
            
                item = re.sub(' +', ' ', item) # remove multiple spaces

                if (' ' in item) and ('Eesti' in item):        
                    split = item.split(' ')
                    return [' '.join(split[1:]), split[0]]
                else: 
                    return [float('nan'), item]
            else:
                return [float('nan'), float('nan')]

        data['city_municipality'] = data.apply(lambda item: extract_city_country(item['country_and_unit'])[0], axis=1)
        data['country'] = data.apply(lambda item: extract_city_country(item['country_and_unit'])[1], axis=1)

        data.drop(columns=['country_and_unit'], inplace=True)
        cols.extend(['country', 'city_municipality'])
    else:
        cols.append('country_and_unit')


    if counter == 3:
        # material
        # to make the following work even for nan values
        data['material'] = data['material'].replace(np.nan, 'nan')
        # prepare single values to be distinguishable
        data['material'] = data['material'].apply(lambda x: x.split('>'))

        # https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list

        mlb = MultiLabelBinarizer()
        data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('material')),
                                columns='material_' + mlb.classes_,
                                index=data.index))

        # technique
        # to make the following work even for nan values
        data['technique'] = data['technique'].replace(np.nan, 'nan')

        # prepare single values to be distinguishable
        data['technique'] = data['technique'].apply(lambda x: x.split('>'))

        mlb = MultiLabelBinarizer()
        data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('technique')),
                                columns='technique_' + mlb.classes_,
                                index=data.index), rsuffix='')
    else:
        cols.extend(['material', 'technique'])

    if counter == 4:
        # location
        data['location_city'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('linn ' in x) else 0)
        data['location_building'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('hoone ' in x) else 0)
        data['location_street'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('tänav ' in x) else 0)
        data['location_country'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('riik ' in x) else 0)
        data['location_address'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('aadress ' in x) else 0)

    if counter == 5:
        ## start - end (formatting)
        #groups the year into its own column with the complete number, if NaN, then 0
        def year_Grouping(x):
            xStr = str(x)
            if xStr == 'nan':
                return 0
            if '.' in xStr:
                xStr = xStr.split('.')
                xStr = xStr[xStr.__len__()-1]
                if xStr.__len__() == 2:
                    xStr = '19' + xStr
                elif xStr == '':
                    return 0
                return int(xStr)
            else:
                return int(xStr)

        # returns 1 if a month is given, 0 if not
        def month_Grouping(x):
            xStr = str(x)
            if '.' in xStr:
                xStr = xStr.split('.')
                if xStr[0].__len__() <= 2 and xStr[1].__len__() <= 2:
                    if xStr[1] == '':
                        return 0
                    elif xStr[1].__len__() == 1:
                        return 1
                    return 1
                xStr = xStr[0]
                if xStr == 'jaan':
                    return 1
                elif xStr == 'veebr':
                    return 1
                if xStr == 'märts':
                    return 1
                elif xStr == 'apr':
                    return 1
                elif xStr == 'mai':
                    return 1
                elif xStr == 'juuni':
                    return 1
                elif xStr == 'juuli':
                    return 1
                elif xStr == 'aug':
                    return 1
                elif xStr == 'sept':
                    return 1
                elif xStr == 'okt':
                    return 1
                elif xStr == 'nov':
                    return 1
                elif xStr == 'dets':
                    return 1
                elif xStr.__len__() == 1:
                    return 1
                return 1
            else:
                return 0

        #returns one if a day is given, 0 if not
        def day_Grouping(x):
            xStr = str(x)
            if '.' in xStr:
                xStr = xStr.split('.')
                if xStr[0].__len__() <= 2 and xStr[1].__len__() <= 2:
                    return 1
                return 0
            else:
                return 0

        def extract_year_from_name(row):
            name = row['name']
            start = row['start']
            if pd.isnull(start) and not pd.isnull(name):
                match = re.search('\d\d\d\d', name)
                if match:
                    start = match.group()
            return start

        data['start'] = data[['name', 'start']].apply(extract_year_from_name, axis=1)    
            
        #grouping applied to the dataframe
        data['startYear'] = data['start'].apply(year_Grouping)
        data['startMonth'] = data['start'].apply(month_Grouping)
        data['startDay'] = data['start'].apply(day_Grouping)
                
        data['endYear'] = data['end'].apply(year_Grouping)
        data['endMonth'] = data['end'].apply(month_Grouping)
        data['endDay'] = data['end'].apply(day_Grouping)

        #if there is no start year, but an end year, then the start year is set to the end year
        for i in range(1,len(data)):
            if data['startYear'].iloc[i] == 0 and data['startDay'].iloc[i] != 0:
                data['startYear'].iloc[i] = data['endYear'].iloc[i]


        #original columns are dropped as they are no longer needed
        data.drop(['start', 'end'], axis=1, inplace=True)
    else:
        cols.extend(['start', 'end'])

    if counter == 6:
        ## color (grouping)
        #Grouping colours by their base colour - to avoid too many extra cloumns when hot encoding -> could always reverse this step
        #by using  something like data['color'] = combined_data_translated['color'] ?

        #The base colours: red, blue, green, grey, yellow, patterned, orange, brown, white, black , pink
        #The most common/distingtive stay unchanged

        def colour_grouping(x):
            if x in ['madara red', 'dark red', 'purple red', 'Red']:
                return 'red'
            elif x in ['light blue', 'dark blue', 'purple blue', 'greenish blue', 'greyish blue']:
                return 'blue'
            elif x in ['light green', 'light olive green', 'grey-green', 'olive green', 'greyish-olive green', 'dark green']:
                return 'green'
            elif x in ['bluish grey', 'dark grey', 'pinkish gray']:
                return 'grey'
            elif x in ['pale yellow', 'light yellow', 'orange-yellow', 'brilliant yellow']:
                return 'yellow'
            elif x in ['brownish orange']:
                return 'orange'
            elif x in ['light brown', 'dark brown', 'greyish brown', 'reddish brown', 'olive brown', 'yellowish brown']:
                return 'brown'
            elif x in ['yellowish white', 'bluish white']:
                return 'white'
            elif x in ['brownish black']:
                return 'black'
            elif x in ['mauve pink']:
                return 'pink'
            elif x in ['<patterned>', 'striped', 'checkered']:
                return 'patterned'
            else:
                return x

        #apply colour_grouping to the dataset
        data['color'] = data['color'].apply(colour_grouping)
        # rescaling numerics values


    # scale numeric features
    if counter == 7:
        data[numeric_features] = MinMaxScaler().fit_transform(data[numeric_features])       

    if counter == 8:
        cols.extend(text_features)
    else:
        data.drop(columns=text_features, inplace=True)

    if counter == 9:
        ## technique - material - sizes (threshold previously encoded)

        # best found combination (local optimum on 500 estimators)
        perc = 0.98
        threshold_sum = len(data) * perc
        min_freq = 7

        tech = col_collection(data, 'technique_')
        mat = col_collection(data, 'material_')
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
    data.drop(columns=['full_nr'], inplace=True)

    # continous numeric features (nan -> 0)
    data[numeric_features] = data[numeric_features].replace(np.nan, 0)

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
    #train, val = train_test_split(train, test_size=0.3, random_state=0)
    test = data.loc[data['source']=='test'].drop('source',axis=1)

    X = train.drop(columns=['type'])
    y = train['type']

    model = RandomForestClassifier(n_estimators=100, random_state=0)

    score = cross_validate(model, X, y, cv=5, scoring={'Acc':"accuracy","F1":"f1_macro"})
    print(counter, score.get('test_Acc').mean(), score.get('test_F1').mean())
    f = open('fe_results.txt', 'a')
    f.write('single ' + str(counter) + ' ' + str(score.get('test_Acc').mean()) + ' ' + str(score.get('test_F1').mean()) + '\n')
    f.close()
