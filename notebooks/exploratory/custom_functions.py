import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# One-hot encode data coming in as a series
# Optional prefix added to each column name
# Optionally input list of columns to use instead ('name_lookup')
def one_hot(srs, prefix='x', name_lookup = False):
    #from sklearn.preprocessing import OneHotEncoder
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


# Extract definitions of encoded naming schemes when given a lookup code
def get_lookups(LUType, df_lookup):
    LUType = str(LUType)
    
    category = df_lookup.loc[df_lookup['LUType'] == LUType].copy()
    category = category.sort_values(by='LUItem')
    result = dict(zip(category.LUItem.str.strip(), category.LUDescription))
    return result