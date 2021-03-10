import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import statsmodels.api as sm
import statsmodels.stats.api as sms

from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder



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
def produce_model(df, x, y, cols_to_clean = [], devct=3, drop_zeros=False):
    model_data = pd.concat([df[y], df[x]], axis=1)
    model_data_trimmed = remove_df_extremes(model_data, cols_to_clean, devct, drop_zeros)
    formula = y + ' ~ ' + '+'.join(x)
    model = ols(formula, model_data_trimmed).fit()
    print('Modeling:', formula, '\n')
    return model, model_data_trimmed


def remove_df_extremes(df, cols_to_clean, devct, drop_zeros=False):
    
    if drop_zeros==True:
        for col in cols_to_clean:
            df = df.loc[df[col]>0].copy()
    
    for col in cols_to_clean:
        df[col] = [float(num) for num in df[col]]
        med = df[col].median()
        std = df[col].std()

        max_ = med + devct*std
        min_ = med - devct*std 

        df[col] = [x if ((x>min_) & (x<max_)) else np.nan for x in df[col]]
    df.dropna(inplace=True)
    return df













def check_assumptions(model, df, y, verbose=True, feature_to_plot=False):
    p = linearity(model, df, verbose, feature_to_plot)
    jb, jb_p = normality(model, df, verbose, feature_to_plot)
    lm, lm_pvalue, fvalue, f_pvalue = homoscedacity(model, df, y, verbose, feature_to_plot)
    vif_avg = independence(model, df, y, verbose, feature_to_plot) 
    
    # return results
    formula = y+'~'+'+'.join(df.drop(y, axis=1).columns)
    
    col_names = ['Formula', 'Linearity p-value', 'Jarque-Bera (JB) metric', 'JB p-value', 'Lagrange multiplier', 'Lagrange multiplier p-value', 'F-score', 'F-score p-value', 'Average VIF']
    data = [formula, p, jb, jb_p, lm, lm_pvalue, fvalue, f_pvalue, vif_avg]
    return pd.DataFrame([data], columns = col_names)

def linearity(model, df, verbose, feature_to_plot):
    lr = linear_rainbow(model)
    p = lr[1]
    
    if verbose == True:
        print('Linearity p-value (where null hypothesis = linear):', p)
        if feature_to_plot != False:
            plt.figure();
            sns.pairplot(df, kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'g'}, 'scatter_kws': {'alpha': 0.3}});
            plt.suptitle('Investigating Linearity', y=1.05);
    return p

def normality(model, df, verbose, plot_feature):
    jb = sms.jarque_bera(model.resid)
    
    if verbose==True:
        print('Normality of Residuals (where null hypothesis = normality): JB stat={}, JB stat p-value={}'.format(jb[0], jb[1]))
    
        if plot_feature != False:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4));

            sm.graphics.qqplot(df[plot_feature], line='45', fit=True, ax=axes[0]);
            plt.suptitle(f'Normality of Residuals: {plot_feature}');

            sns.distplot(model.resid, label='Residuals', ax=axes[1])
            sns.distplot(np.random.normal(size=10000), label='Normal Distribution', ax=axes[1])
            axes[1].legend()
    return jb[0], jb[1]    

def homoscedacity(model, df, y, verbose, plot_feature):
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    
    if verbose==True:
        print("Homoscedacity (where null hypothesis = homoscedastic): lagrange p-value={} and f-value's p-value={}".format(lm_pvalue, f_pvalue))
    
        if plot_feature != False:
            predicted = model.predict()
            error = df[y] - predicted
            plt.figure();
            plt.scatter(df[plot_feature], error, alpha=0.3);
            plt.plot([df[plot_feature].min(), df[plot_feature].max()], [0,0], color='black')
            plt.xlabel(plot_feature)
            plt.ylabel("Error (Actual-Predicted)")
            plt.title('Homoscedacity of Residuals');
    return lm, lm_pvalue, fvalue, f_pvalue

# CITATION: function content taken from Flatiron School Study Group material
def independence(model, df, y, verbose, plot_feature):
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
            print('\nVariance Inflation Factors:\n', df_vif)
    
    if verbose == True:
        if plot_feature != False:
            CorrMatrix = df.corr()
            plt.figure();
            sns.heatmap(CorrMatrix, annot=True);
    
    if type(df_vif)==type('test'):
        return 'NA'
    else:
        return df_vif.VIF.mean()