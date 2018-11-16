import numpy as np
import pandas as pd
import pywt
import requests

# to rates
def to_rate(df):
    pass
    # start_days = df.values[::5, :]
    # end_days = df.values[4::5, :]
    # rates = (end_days - start_days) / start_days

# filling  
def linspace_fill(array, fill_last=True):
    arr = array.copy()
    # index of first value not nan
    j = np.argmin(np.isnan(arr))
    arr[:j] = arr[j]
    filling, ind_before_na = False, None
    for i, isnan in enumerate(np.isnan(arr)):
        if ( filling == False ) and isnan:
            filling = True
            ind_before_na = i - 1
        elif ( filling == True ) and not isnan:
            arr[ind_before_na:i+1] = np.linspace(arr[ind_before_na], arr[i], i+1-ind_before_na)
            filling = False
    if filling and fill_last:
        arr[ind_before_na:] = arr[ind_before_na]
    return arr

# spliting
def split_data(dataframe, nweeks, target_len):
    if type(dataframe) == pd.DataFrame:
        array = dataframe.values
    else:
        array = dataframe
    total_len = nweeks + target_len
    M = np.array( [ array[i:i+total_len, :] for i in range(array.shape[0]-total_len+1) ] )
    M = np.array([ M[:, :, i] for i in range(array.shape[1])])
    return M[:, :, :nweeks], M[:, :, nweeks:]

def train_test_split(dataframe, nweeks, target_len, test_size):
    '''Returns
    -------
    trainX, trainy, testX, testy
    '''
    k = (target_len - 1) + test_size
    X, y = split_data(dataframe, nweeks, target_len)
    
    return X[:, :-k, :], \
           y[:, :-k, :], \
           X[:, -test_size:, :], \
           y[:, -test_size:, :]

# merge original data with new data
def merge(df1, df2):
    '''merge two dataframes with different shape.
    Index and columns are preserved.
    df1 have higher priority'''
    df = pd.DataFrame( index=df1.index.union(df2.index), 
                       columns=df1.columns.union(df.columns) )
    d2 = df.copy()
    df.loc[df1.index, df1.columns] = df1
    d2.loc[df2.index, df2.columns] = df2
    isnull = df.isna()
    df[isnull] = d2[isnull]
    return df

# scaling
def min_max(df, min_min=True):
    '''scaling raw data'''
    if min_min:
        column_min = df.min(axis=0)
        df = df - column_min
    column_max = df.max(axis=0)
    df = df / column_max
    return df

# web crawler ...
def get_data(coid_name, start='2018/11/5', end='2018/11/9', error_string='µL¸ê®Æ',
            url_format='http://esunbankfintest.moneydj.com/W4/bcd/BCDNavList.djbcd?a=%s&B=%s&C=%s'):
    
    url = url_format%(coid_name, start, end)
    text = requests.get(url).text
    
    if text == error_string:
        return {}
    else:
        dates, values = text.split(' ')
        return dict( zip(dates.split(','), values.split(',')) )

def get_datas_frame(coid_names, **kwargs):
    '''
    `coid names` : iterable,  list of coid names
    `start`, `end` : str,  "2018/11/8"
    `error_string` : str, default "µL¸ê®Æ"
    `url_format`...
    '''
    df = pd.DataFrame( { coid: get_data(coid, **kwargs) for coid in coid_names } )
    df.index = pd.DatetimeIndex(df.index)
    df = df.astype(float)
    print('number of missing values:', df.isna().values.sum())
    return df

# wavelet transform
def transform(df, path='', save=True):
    org_mean = df.mean(axis=0).values.reshape(1, -1)

    for colname in df:
        df[colname] = wavelet_tansform(df[colname].values)

    denoised_mean = df.mean(axis=0).values.reshape(1, -1)
    shifted = df.values - denoised_mean + org_mean
    df = pd.DataFrame(shifted, columns=df.columns)
    if save:
        df.to_csv(path, index=False)
    return df

def wavelet_tansform(raw):
    (ca, cd) = pywt.dwt(raw, "haar")                
    cat = pywt.threshold(ca, np.std(ca), mode="soft")                
    cdt = pywt.threshold(cd, np.std(cd), mode="soft")               
    trans_raw = pywt.idwt(cat, cdt, "haar")
    if np.isnan(trans_raw).any():
        return raw
    return trans_raw
