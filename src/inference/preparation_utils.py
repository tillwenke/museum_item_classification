import numpy as np
import pandas as pd
import re

text_features = ['name', 'commentary', 'text', 'legend', 'initial_info', 'additional_text', 'damages']

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


def extract_value(value, present):
    if present == 1:
        return value
    else:
        return 0

# make all size features numeric
def extract_height_width(item):    
    if item == 0:
        return [0.0,0.0]
    else:        
        return [float(i) for i in item]

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

## event_type (brackets)
def strip_brackets(x):
    x = str(x)
    x = x.strip('< >')
    if '\u200b' in x:
        x = x.replace('\u200b', '')
    return x

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

# collects all column names that start with col_start
def col_collection(data, col_start):
        cols = []
        for c in data.columns:
            if (c.startswith(col_start)):
                cols.append(c)
        return cols

def collect_text(item, tf):
    return ' '.join(item[tf]).strip()

def get_text_df(data):
    
    tf = [value for value in text_features if value in data.columns]
    data[tf] = data[tf].fillna('')
    data['text_features'] = data.apply(lambda item: collect_text(item, tf),axis=1)

    final_features = [value for value in ['text_features','type','source'] if value in data.columns]
    text = data[final_features]

    text.text_features = text.text_features.apply(lambda x: x.strip())
    text = text[text.text_features != '']
    return text

