# Classification of items in Estonian Museums
Part of this kaggle challenge: https://www.kaggle.com/competitions/caps-in-museums-data-part-2

For collaboration: https://docs.google.com/document/d/1UM7QEFyb16v7Zd-cRcEMz8tcKqE91UEl0C2PXt6B85A/edit?usp=sharing

Column discovery: https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing

# Description of the features
SYNDMUSE_LIST - usually of three types - a preparation event, a theme event and a collection event. A themed event is then all those photoshoots, theatre, music, funerals, etc etc. It is a categorical cell

LOCATION - consists of two parts: the first word is a location specifier (selectable from the drop-down menu), the second is free text as a place name.

START and END are a numeric value up to the exact date

ENNE_KR is a no/yes value.

the MONTH is somehow left in

COUNTRY_ADMIN_COUNTRY - for foreign countries only the country name, for Estonia a more specific administrative unit may be added.

PARISH - is an Estonian parish.

PARTICIPANT - name of the person or organisation

PARTICIPANT_ROLE - mandatory field if PARTICIPANT selected, specifies the role of the participant in the event (e.g. author, associate, sitter, photographer, etc.).

NAME - name of the museum object

TECHNIQUE - what technique was used to make the object (e.g. copying, ceramics, black and white photography).

COMMENTARY - free text commentary on the technique

Table of dimensions

PARAMETER - predefined formats (e.g. width, number of pages)

UNIT - unit of measurement (e.g. page, cm)

VALUE - usually a numeric field

COMMENTARY - comment on dimensions

MATERIAL - what the specimen is made of (e.g. paper, wood, gold).

COMMENTARY - comment on the material

SITUATION - good, bad, very bad, unspecified, satisfactory

DAMAGE - free text box to specify damage (e.g. torn, crack, etc.)

TEXT - additional name of the museum object as free text.

TYPE - type of additional name, choice (e.g. former name, document name).

ADDITIONAL TEXT - free text description of the museum object.

APPENDIX - type of text, choice (e.g. text on object, comment, physical description, etc.).
ACR - Abbreviation of museum, text field.

TRT - Tulmera Book Symbol, text field.

TRS - Tulip number series number, numeric field.

TRJ - TIN serial number, numeric field.

TRL - Supplementary TIN number, text field.

KT - Whole identifier, text field

KS - Whole series number, numeric field

KJ - Whole serial number, numeric field.

KL - Whole extension number, text field.


# How to use:

Run the notebooks in the following order:

### Data discovery:

data_discovery

### Data Preparation:

combine_test_and_train
translation
whitespaces
unit_sizes
split_content_feature
formatting_1
formatting_2
hot_encoding_prep - upto Re-hot encoding 
adding_and_deleting_columns
prep_final

To adjusting thresholds or to remove extra columns later:
hot_encoding_prep - from Re-hot encoding 
adding_and_deleting_columns


### Modelling: 

prepare for models:
rebalancing

Running models - here order in not as important 

random_forest
random_forest_balanced
crossval_randomforest
neural_networks
nlp
predictions
xgboost_tries
xgboost_tries_balanced
voter

The details of these notebooks are found in README_data_prep and README_modelling respectively 
