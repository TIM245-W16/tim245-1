import numpy as no
import pandas as pd
import dateutil.parser as pr

def load_currency(currency, column=None):
    raw_data = get_raw(currency)
    idx = raw_data.loc[:,'time'].apply(lambda x: pr.parse(x[:11]))

    with_idx = pd.DataFrame(raw_data.loc[:,valid_columns()], columns=valid_columns())

    #with_idx = raw_data.set_index(['time']).drop(['complete','volume'], axis=1)
    with_idx = raw_data.drop(['time','complete','volume'], axis=1)

    if(column == None):
        return with_idx
    else:
        return with_idx[column]

def load_test_train(currency):
    raw_data = get_raw(currency).dropna()

    # test is going to be the most recent year (last 252 periods)
    test_data = raw_data.iloc[-252:]
    
    # we'll use the three years before to build a model (756)
    train_data = raw_data.iloc[-1008:-252]
    
    check = raw_data.iloc[-756:]

    return [train_data,test_data]

def valid_columns():
    return ['closeMid','highMid','lowMid','openMid','complete','volume']

def valid_currencies():
    return ['EUR_CHF','EUR_GBP','EUR_NZD','GBP_AUD','GBP_JPY','NZD_USD']

def get_raw(currency, column='closeMid'):
    return pd.read_json('./data/{}.txt'.format(currency))[column]