from setup_general import *
from prep_helpers import *

source = 'data/typeless/AM_ETMM.csv'
sink = 'data/inference/AM_ETMM.csv'

#############################

data = pd.read_csv(source, index_col='id')
# Feature specific engineering
## units - sizes -values
# Finish unit translation/ unification &  values to float
data['value'] = data['value'].apply(lambda x: float(x.replace(',', '.')) if type(x) == str else x)

# unify units
data['unit'] = data['unit'].replace('10 x 15 cm','100 x 150 mm')

# mm to cm
data['value'] = data.apply(lambda item: item['value'] / 10 if item['unit'] == 'mm' else item['value'], axis=1)
data['unit'] = data['unit'].replace('mm','cm')
data['value'] = pd.to_numeric(data['value'])

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

#  for all new "parameter with unit" columns put the value in the column where a 1 is - others are 0 and remain 0
for column in data.columns:
    if ' IN ' in column and '*' not in column:
        data[column] = data.apply(lambda item: extract_value(item['value'], item[column]), axis=1)        

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

data['country_and_unit'] = data.apply(lambda x: empty_to_nan(x['country_and_unit']), axis=1)
data['technique'] = data['technique'].apply(lambda x: x.strip() if (type(x) == str) else x)
## country_unit - material - technique - location (splitting for features including multiple information)

data['city_municipality'] = data.apply(lambda item: extract_city_country(item['country_and_unit'])[0], axis=1)
data['country'] = data.apply(lambda item: extract_city_country(item['country_and_unit'])[1], axis=1)

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

# location
data['location_city'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('linn ' in x) else 0)
data['location_building'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('hoone ' in x) else 0)
data['location_street'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('tänav ' in x) else 0)
data['location_country'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('riik ' in x) else 0)
data['location_address'] = data['location'].apply(lambda x: 1 if (type(x) == str) and ('aadress ' in x) else 0)

## start - end (formatting)
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

data['event_type'] = data['event_type'].apply(strip_brackets)
## color (grouping)
#Grouping colours by their base colour - to avoid too many extra cloumns when hot encoding -> could always reverse this step
#by using  something like data['color'] = combined_data_translated['color'] ?



#apply colour_grouping to the dataset
if 'color' in data.columns:
    data['color'] = data['color'].apply(colour_grouping)
# rescaling numerics values

# scale numeric features
numeric_features = [value for value in ['ks', 'musealia_seria_nr', 'musealia_queue_nr', 'collection_queue_nr', 'element_count'] if value in data.columns]
# continous numeric features (nan -> 0)

data[numeric_features] = data[numeric_features].replace(np.nan, 0)

data[numeric_features] = MinMaxScaler().fit_transform(data[numeric_features])
# get typeless ready
cols = ['musealia_additional_nr', 'collection_mark', 'musealia_mark', 'museum_abbr', 'before_Christ', 'is_original', 'class', 'state', 'event_type', 'participants_role', 'parish', 'color', 'collection_additional_nr', 'damages', 'participant', 'location', 'name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text', 'country', 'city_municipality']
columns = [value for value in cols if value in data.columns]
data = pd.get_dummies(data, columns=columns)
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

data.to_csv(sink, index=True)