import pandas as pd

def one_hot(srs, prefix='x', name_lookup = False):
    from sklearn.preprocessing import OneHotEncoder
    #import pandas as pd
    
    ohe = OneHotEncoder(sparse=False, drop='first')
    
    df = pd.DataFrame(srs)
    df = pd.DataFrame(ohe.fit_transform(df))
    
    if name_lookup == False:    
        names = ohe.get_feature_names()
        new_names = [prefix+'_'+x[3:] for x in names]
        df.columns = new_names
    else:
        names = ohe.get_feature_names()
        codes = [x[3:] for x in names]
        new_names = [name_lookup[x] for x in codes]
        df.columns = new_names
    
    for col in df.columns:
        df[col] = df[col].astype('int')
    
    return df