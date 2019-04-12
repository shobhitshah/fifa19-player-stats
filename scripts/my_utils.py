import numpy as np

def clean_column_names(train):
    return train.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

def preprocess_columns(train):
    train=train.drop(columns=['photo', 'flag', 'club_logo'])

    train['value_unit'] = train['value'].str[-1]
    train['nvalue'] = np.where(train['value_unit'] == '0', 0,
                              train['value'].str[1:-1].replace(r'[a-zA-Z]',''))
    train['nvalue'] = train['nvalue'].astype(float)
    train['nvalue'] = np.where(train['value_unit'] == 'M', train['nvalue'], train['nvalue'] / 1000)

    train['wage_unit'] = train['wage'].str[-1]
    train['nwage'] = np.where(train['wage_unit'] == '0', 0,
                              train['wage'].str[1:-1].replace(r'[a-zA-Z]',''))
    train['nwage'] = train['nwage'].astype(float)
    train['nwage'] = np.where(train['wage_unit'] == 'K', train['nwage'], train['nwage'] * 1000)

    return train
