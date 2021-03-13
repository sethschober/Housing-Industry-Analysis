import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf

from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression





def elimination_by_code(series, code_to_keep):
    series.loc[series != code_to_keep] = np.nan
    return series









def identify_latest_sale(docdates, parcel_ids):
    latest_parcel_sale = []
    data = pd.DataFrame([docdates, parcel_ids]).T
    data.DocumentDate = data.DocumentDate.astype('datetime64')
 
    for i, parcel_id in enumerate(data.Parcel_ID):
        relevant_docdates = data.loc[data.Parcel_ID == parcel_id, 'DocumentDate']
        max_docdate = relevant_docdates.values.max()
        this_datetime = np.datetime64(data.iloc[i, 0]) 
        latest_parcel_sale.append(this_datetime == max_docdate)
    return latest_parcel_sale




def avg_price_for_duped_parcels(data):
    dupes = data.loc[data.SaleCount > 1]
    for i, ind in enumerate(dupes.index):
        parcel_id = data.loc[ind, 'Parcel_ID']
        parcels_w_parcel_id = data.loc[data.Parcel_ID == parcel_id, 'SalePrice']

        avg_price_for_id = parcels_w_parcel_id.values.mean()
        for parcel_index in parcels_w_parcel_id.index:
            data.at[parcel_index, 'SalePrice'] = avg_price_for_id
    return data





# Extract definitions of encoded naming schemes when given a lookup code
def get_lookups(LUType):
    conn = sqlite3.connect('../../data/processed/main.db')
    lookups_query = '''SELECT * FROM lookups'''
    df_lookup = pd.read_sql(lookups_query, conn)
    conn.close()
    
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


def cut_extremes(series, n):
    std = series.std()
    median = series.median()
    max_ = median + n*std
    min_ = median - n*std
    
    cut = lambda x: np.nan if x<min_ else np.nan if x>max_ else x
    return series.apply(cut)




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

    # Determine which features were removed
    if verbose==True:
        removed = list(X.columns)
        for item in included:
            print(item)
            removed.remove(item)

    print('Remaining features:', included)
    print('Removed features:', removed)
    
    return included









    
    
    
# CREATES MODEL
def produce_model(df, x, y):
    formula = y + ' ~ ' + '+'.join(x)
    model = ols(formula, df).fit()
    print('Modeling:', formula)
    return model, df[[y]+x].copy()












def check_assumptions(model, df, y, verbose=False, feature_to_plot=False):
    p        = linearity(model, df, verbose, feature_to_plot)
    jb, jb_p = normality(model, df, verbose, feature_to_plot)
    lm, lm_pvalue, fvalue, f_pvalue = homoscedacity(model, df, y, verbose, feature_to_plot)
    vif_avg  = independence(model, df, y, verbose, feature_to_plot) 
    
    x = '+'.join(df.drop(y, axis=1).columns)
    r2_adj = model.rsquared_adj
    col_names = ['Y', 'X', 'Linearity p-value', 'Jarque-Bera (JB) metric', 'JB p-value', 'Lagrange multiplier', 'Lagrange multiplier p-value', 'F-score', 'F-score p-value', 'Average VIF', 'R^2 (Adj.)']
    data = [y, x, p, jb, jb_p, lm, lm_pvalue, fvalue, f_pvalue, vif_avg, r2_adj]
    return pd.DataFrame([data], columns = col_names)

def linearity(model, df, verbose, feature_to_plot):
    p = linear_rainbow(model)[1]
    
    if verbose == True:
        print('Linearity p-value (where null hypothesis = linear):', p)
    if feature_to_plot != False:
        # Identify non-categorical features
        df_plotter = df.copy()
        for col in df_plotter.columns:
            if df_plotter[col].value_counts().shape[0] == 2:
                df_plotter.drop(col, axis=1, inplace=True)
        plt.figure();
        sns.pairplot(df_plotter, kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'g'}, 'scatter_kws': {'alpha': 0.3}});
        plt.suptitle('Investigating Linearity', y=1.05);
    return p

def normality(model, df, verbose, feature_to_plot):
    jb = sms.jarque_bera(model.resid)
    
    if verbose==True:
        print('Normality of Residuals (where null hypothesis = normality): JB stat={}, JB stat p-value={}'.format(jb[0], jb[1]))
    
    if feature_to_plot != False:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4));

        sm.graphics.qqplot(df[feature_to_plot], line='45', fit=True, ax=axes[0]);
        plt.suptitle(f'Normality of Residuals: {feature_to_plot}');

        sns.distplot(model.resid, label='Residuals', ax=axes[1])
        sns.distplot(np.random.normal(size=10000), label='Normal Distribution', ax=axes[1])
        axes[1].legend()
    return jb[0], jb[1]    

def homoscedacity(model, df, y, verbose, feature_to_plot):
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    
    if verbose==True:
        print("Homoscedacity (where null hypothesis = homoscedastic): lagrange p-value={} and f-value's p-value={}".format(lm_pvalue, f_pvalue))
    
    if feature_to_plot != False:
        predicted = model.predict()
        error = df[y] - predicted
        plt.figure();
        plt.scatter(df[feature_to_plot], error, alpha=0.3);
        plt.plot([df[feature_to_plot].min(), df[feature_to_plot].max()], [0,0], color='black')
        plt.xlabel(feature_to_plot)
        plt.ylabel("Error (Actual-Predicted)")
        plt.title('Homoscedacity of Residuals');
    return lm, lm_pvalue, fvalue, f_pvalue



def independence(model, df, y, verbose, feature_to_plot):
    features = df.drop(y, axis=1).columns
    if len(features) == 1:
        df_vif='NA (single variable)'
        if verbose==True:
            print('Variance Inflation Factors:', df_vif)
    else:
        df_vif = pd.DataFrame()
        df_vif['Feature'] = features
        df_vif['VIF'] = [variance_inflation_factor(df.drop(y, axis=1).values, i) for i in range (len(features))]
        if verbose==True:
            print('Variance Inflation Factors:\n', df_vif)
    
    if feature_to_plot != False:
        CorrMatrix = df.corr()
        plt.figure();
        sns.heatmap(CorrMatrix, annot=True);
    
    if type(df_vif)==type('test'):
        return 'NA'
    else:
        return df_vif.VIF.mean()
    
    
    
    
    
##### FORWARD SELECTION #####
#SOURCE: https://planspace.org/20150423-forward_selection_with_statsmodels/
def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model










def recursive_feature_elimination(x, y, n):
    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select = n)
    selector = selector.fit(x, y.values.ravel())
    top_features = x.loc[:, selector.support_]
    
#     # Determine which features were removed
#     removed = list(x.columns)
#     for item in keepers:
#         removed.remove(item)

#     # print('\nRemaining features:', keepers)
#     # print('\nRemoved features:', removed)
    
    return top_features