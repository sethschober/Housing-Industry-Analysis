import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder
from statsmodels.formula.api import ols


# One-hot encode data coming in as a series
# Optional prefix added to each column name
# Optionally input list of columns to use instead ('name_lookup')
from sklearn.preprocessing import OneHotEncoder
def one_hot(srs, prefix=False, name_lookup = False):
    
    ohe = OneHotEncoder(sparse=False, drop='first')
    
    df = pd.DataFrame(srs)
    df = pd.DataFrame(ohe.fit_transform(df))
    
    if name_lookup == False:    
        names = ohe.get_feature_names()
        new_names = [prefix+'_'+x[3:] for x in names]
        df.columns = new_names    
    elif prefix == False:
        names = ohe.get_feature_names()
        codes = [x[3:] for x in names]
        new_names = [name_lookup[x] for x in codes]
        df.columns = new_names
    else:
        names = ohe.get_feature_names()
        #new_names = [prefix+'_'+x[3:] for x in names]
        #df.columns = new_names   

        codes = [x[3:] for x in names]
        new_names = [prefix+'_'+name_lookup[x] for x in codes]
        df.columns = new_names
    
    for col in df.columns:
        df[col] = df[col].astype('int')
    
    df.columns = [x.replace(' ','').replace('-','') for x in df.columns]
    
    return df


# Extract definitions of encoded naming schemes when given a lookup code
def get_lookups(LUType, df_lookup):
    LUType = str(LUType)
    
    category = df_lookup.loc[df_lookup['LUType'] == LUType].copy()
    category = category.sort_values(by='LUItem')
    result = dict(zip(category.LUItem.str.strip(), category.LUDescription))
    return result


# Strip leading and trailing spaces
def strip_spaces(df):
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


# Remove *x* std deviations of data from an input
def remove_extremes(df, col, devct):
    df[col] = [float(num) for num in df[col]]
    #cleaned = data.loc[data>0].copy()

    std = cleaned.std()
    med = cleaned.median()

    cleaned = cleaned.loc[(cleaned > (med - std*devct)) & (cleaned < (med+std*devct))].copy() 
    return cleaned






# FUNCTION TAKEN DIRECTLY FROM FLATIRON COURSE MATERIAL (https://github.com/learn-co-curriculum/dsc-model-fit-linear-regression-lab)
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in =  0.0499, 
                       threshold_out = 0.0500, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            worst_feature = pvalues.index[worst_feature] # SETH ADDED FROM ORIGINAL TO RESOLVE RUNTIME ERRORS
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



def remove_df_extremes(df, devct, drop_zeros=False):
    
    if drop_zeros==True:
        for col in df.columns:
            df = df.loc[df[col]>0].copy()
    
    for col in df.columns:
        df[col] = [float(num) for num in df[col]]
        med = df[col].median()
        std = df[col].std()

        max_ = med + devct*std
        min_ = med - devct*std 

        df[col] = [x if ((x>min_) & (x<max_)) else np.nan for x in df[col]]
    df.dropna(inplace=True)
    return df

# Create QQ-plot
def qq(df, col):
    plt.figure(figsize=(12,6));
    sm.graphics.qqplot(df[col], line='45', fit=True)
    plt.title(f'Normality Assumption Check: QQ plot of {col} values');
    
# Create DistPlot
def dist(df, x):
    plt.figure(figsize=(12,6));
    sns.distplot(df[x])
    plt.title(f'Distribution of {x} (KDE)')
    
# Create scatterplot (lmplot)
def scatter(df, x, model):
    plt.figure(figsize=(12, 6));
    sns.lmplot(data = df, x=x, y=y, line_kws={'color':'r'})    
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Linearity Assumption: {x} vs. {y}');
    
    
    
# ESSENTIAL FUNCTION: CREATES MODEL
def produce_model(df, x, y, devct=3, drop_zeros=False):
    model_data = pd.concat([df[y], df[x]], axis=1)
    model_data_trimmed = remove_df_extremes(model_data, devct, drop_zeros)
    formula = y + ' ~ ' + '+'.join(x)
    model = ols(formula, model_data_trimmed).fit()
    print('Modeling:', formula, '\n')
    return model, model_data_trimmed