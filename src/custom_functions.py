import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import os, sys

from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression



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




def elimination_by_code(series, code_to_keep):
    """Filters a Series by setting all values other than code_to_keep to np.nan

    This is used when simplifying data imported from King Couunty dataset, which provides irrelevant data that needs to be eliminated. By setting invalid values to np.nan, the 'na' values can later be dropped from the DataFrame

    Parameters:
    -----------
    series : Pandas Series containing the data to be filtered
    code_to_keep : the only value in series that is to be retained. Others are replaced by np.nan
    
    Returns:
    --------
    series : the input data after being filtered to only code_to_keep values
    """
    
    series.loc[series != code_to_keep] = np.nan
    return series









def identify_latest_sale(docdates, parcel_ids):
    """Determine if a sale is the most recent sale for each parcel

    The function returns a True/False series indicating whether the timestamp associated with each row is the most recent for each parcel.
    
    Parameters:
    -----------
    docdates : series containing the dates to compare, corresponding to the parcel_ids
    parcel_ids : series containing the parcel_id associated with each docdate
    
    Returns:
    --------
    latest_parcel_sale : True/False series showing whether the docdate is the most recent date for each parcel_id
    
    """
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
    """Calculate the average price for each parcel
    
    Parameters:
    -----------
    data : DataFrame containing at least two columns: SalePrice and Parcel_ID
    
    Returns:
    --------
    data : input's SalePrice column is replaced with the average of SalePrice *for that Parcel_ID*

    """
    
    # Identify which Parcel_IDs have more than one row 
    dupes = data.loc[data.SaleCount > 1]
    
    # Loop over all duplicates
    for i, ind in enumerate(dupes.index):
        
        # Select data corresponding to that Parcel_ID
        parcel_id = data.loc[ind, 'Parcel_ID']
        parcels_w_parcel_id = data.loc[data.Parcel_ID == parcel_id, 'SalePrice']

        # Calculate average price for that parcel
        avg_price_for_id = parcels_w_parcel_id.values.mean()
        
        # Loop over all parcels with that Parcel_ID and assign the average price
        for parcel_index in parcels_w_parcel_id.index:
            data.at[parcel_index, 'SalePrice'] = avg_price_for_id
    return data





def get_lookups(LUType):
    """Extract definitions of King County lookup code by value

    Parameters:
    -----------
    LUType : code corresponding to the data dictionary item to look up
    
    Returns:
    --------
    result : dictionary with the King County lookup code as index and corresponding definition as value
    """
    
    # Import lookup database
    df_lookup = pd.read_csv(os.path.join('..','..', 'data', 'raw', 'EXTR_LookUp.csv'), dtype='str')
    df_lookup = strip_spaces(df_lookup)
    
    # Convert input to string since that is how it is stored in dictionary
    LUType = str(LUType)
    
    # Isolate the definitions for the provided code and store in dictionary
    category = df_lookup.loc[df_lookup['LUType'] == LUType].copy()
    category = category.sort_values(by='LUItem')
    result = dict(zip(category.LUItem.str.strip(), category.LUDescription))
    return result





def strip_spaces(df):
    """Strip spaces for every single cell in an entire DataFrame

    Parameters:
    -----------
    df : DataFrame of any size with *all values formatted as text*
    
    Returns:
    --------
    df : original DataFrame stripped of leading and trailing spaces
    """
    
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df




def cut_extremes(series, n):
    """Remove outliers outside of n std deviations from median
    
    Parameters:
    -----------
    series : numerical series to be trimmed at n std deviations
    n : the number of standard deviations to include in the output
    
    Returns:
    --------
    series : original series with outliers replaced with np.nan
    """
    
    # Calculate std deviation, median, and corresponding max/min range
    std = series.std()
    median = series.median()
    max_ = median + n*std
    min_ = median - n*std
    
    # Remove values outside of max and min range
    cut = lambda x: np.nan if x<min_ else np.nan if x>max_ else x
    
    # Apply function and return
    return series.apply(cut)





def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in =  0.0499, 
                       threshold_out = 0.0500, 
                       verbose=True):
    """ 
    Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    
    SOURCE: FUNCTION TAKEN DIRECTLY FROM FLATIRON COURSE MATERIAL (https://github.com/learn-co-curriculum/dsc-model-fit-linear-regression-lab) with minor edits    
    
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

    # Determine which features were removed and print results
    if verbose==True:
        removed = list(X.columns)
        for item in included:
            print(item)
            removed.remove(item)
        print('Remaining features:', included)
        print('Removed features:', removed)
    return included




def produce_model(df, x, y, verbose=True):
    """Create linear regression model

    Parameters:
    -----------
    df : Pandas DataFrame used a source for model. *Must* include the columns provided as x and y
    x : List of column names to include as features in model
    y : String corresponding to the column name of the model output variable
    
    Returns:
    --------
    Model : StatsModels OLS linear regression model object
    df : DataFrame containing the columns provided in x and y, the data used dto create the model
    """
    
    # Define formula for writing OLS model
    formula = y + ' ~ ' + '+'.join(x)
    model = ols(formula, df).fit()
    
    # Tell the user what the model is
    if verbose:
        print('Modeling:', formula)
    
    # Place columns corresponding to y and x into a DataFrame to return
    df_model = df[[y]+x].copy()
    return model, df_model







def check_assumptions(model, df, y, verbose=False, feature_to_plot=False):
    """Check the assumptions of linear regression validity

    Essential in determining the validity of a linear regression model by providing information about the four key assumptions in order to evaluate model performance: linearity, normality of residuals, homoscedacity, and independence. Optionally provide graphical representations annd/or printouts of key metrics, in addition to returning a DataFrame with those metrics
    
     Parameters: 
     -----------
     model : statsmodels model object based on df with the output set as y
     df : DataFrame used to create the 'model' input
     y : the dependent variable of the model. Must be included in df and used to create model.
     verbose : optionally print out metrics as calculations are made. These metrics are also returned
     feature_to_plot : select one column from df for which to plot the assumptions. Some graphs inherently show all features of the model, while others can only display one feature. In the latter case, this is the feature that is used. If a feature is not selected, no graphs are displayed.
    
    Returns:
    --------
    performance_metrics : DataFrame with one row containing: Model Output (y), Model Inputs (x), Linearity p-value, JB metric, JB p-value, Lagrange multiplier, Lagrange p-value, F-score, F-score p-value, Average VIF, Adjusted Rsquared    
    """
    
    # Run four functions to evaluate each of the assumptions
    p        = linearity(model, df, verbose, feature_to_plot);
    jb, jb_p = normality(model, df, verbose, feature_to_plot);
    lm, lm_pvalue, fvalue, f_pvalue = homoscedacity(model, df, y, verbose, feature_to_plot);
    vif_avg  = independence(model, df, y, verbose, feature_to_plot);
    
    # Assemble and calculate variables to add to performance metrics
    # inputs formula
    x = '+'.join(df.drop(y, axis=1).columns)
    
    # adjusted r_squared
    r2_adj = model.rsquared_adj
    
    # DataFrame to consolidate all performance metrics
    col_names = ['Y', 'X', 'Linearity p-value', 'Jarque-Bera (JB) metric', 'JB p-value', 'Lagrange multiplier', 'Lagrange multiplier p-value', 'F-score', 'F-score p-value', 'Average VIF', 'R^2 (Adj.)']
    data = [y, x, p, jb, jb_p, lm, lm_pvalue, fvalue, f_pvalue, vif_avg, r2_adj]
    performance_metrics = pd.DataFrame([data], columns = col_names)
    
    return performance_metrics


def linearity(model, df, verbose, feature_to_plot):
    """Evaluate linearity assumption of linear regression
    
    Parameters:
    -----------
    model : StatsModels model object based on df
    df : data used to create model
    verbose : optionally print metrics to console
    feature_to_plot : optionally display graph to inspect linearity visually
    
    Returns:
    --------
    p : p-value from the linear_rainbow test, serving as a linearity metric
    
    """
    
    # Calculate p with linear_rainbow
    p = linear_rainbow(model)[1]
    
    # Optionally print output
    if verbose == True:
        print('Linearity p-value (where null hypothesis = linear):', p)
    
    # Optionally plot linearity graphically
    if feature_to_plot != False:
        
        # Identify non-categorical features (which are excluded)
        df_plotter = df.copy()
        
        # loop over all dataframe columns
        for col in df_plotter.columns:
            
            # remove all features with only 2 unique values (binary / one-hot enoded, for which graphical linearity is less relevant)
            if df_plotter[col].value_counts().shape[0] == 2:
                df_plotter.drop(col, axis=1, inplace=True)
        
        # Plot results, as desired
        plt.figure();
        sns.pairplot(df_plotter, kind='reg', diag_kind='kde', plot_kws={'line_kws':{'color':'g'}, 'scatter_kws': {'alpha': 0.3}});
        plt.suptitle('Investigating Linearity (Continuous Features Only)', y=1.05);
    
    return p;



def normality(model, df, verbose, feature_to_plot):
    """Evaluate residual normality assumption of linear regression
    
    Parameters:
    -----------
    model : StatsModels model object based on df
    df : data used to create model
    verbose : optionally print metrics to console
    feature_to_plot : optionally display graph of residuals for selected feature
    
    Returns:
    --------
    jb[0] : JB metric
    jb[1] : p-value associated with JB metric
    
    """
    
    # Run Jarque-Bera test
    jb = sms.jarque_bera(model.resid)
    
    # Optionally print findings to console
    if verbose==True:
        print('Normality of Residuals (where null hypothesis = normality): JB stat={}, JB stat p-value={}'.format(jb[0], jb[1]))
    
    # Optionally plot residuals in two formats
    if feature_to_plot != False:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4));

        # QQ-plot
        sm.graphics.qqplot(df[feature_to_plot], line='45', fit=True, ax=axes[0]);
        plt.suptitle(f'Normality of Residuals: {feature_to_plot}');

        # Distribution plot compared to normal distribution
        sns.distplot(model.resid, label='Residuals', ax=axes[1])
        sns.distplot(np.random.normal(size=10000), label='Normal Distribution', ax=axes[1])
        axes[1].legend()
    
    return jb[0], jb[1]    




def homoscedacity(model, df, y, verbose, feature_to_plot):
    """Evaluate homoscedacity assumption of linear regression
    
    Parameters:
    -----------
    model : StatsModels model object based on df
    df : data used to create model
    verbose : optionally print metrics to console
    feature_to_plot : optionally display graph of residuals for selected feature
    
    Returns:
    --------
    lm : Lagrange multiplier
    lm_pvalue : p-value associated with Lagrange multiplier
    fvalue : F-statistic
    f_pvalue : p-value associated with F-statistic
    
    """
    
    # Run Breuschpagan test to calculate performance metrics
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
    
    # Optionally print results to console
    if verbose==True:
        print("Homoscedacity (where null hypothesis = homoscedastic): lagrange p-value={} and f-value's p-value={}".format(lm_pvalue, f_pvalue))
    
    # Optionally plot error / residuals
    if feature_to_plot != False:
        # Calculate predicted values and error
        predicted = model.predict()
        error = df[y] - predicted
        
        # Plot
        plt.figure();
        plt.scatter(df[feature_to_plot], error, alpha=0.3);
        plt.plot([df[feature_to_plot].min(), df[feature_to_plot].max()], [0,0], color='black')
        plt.xlabel(feature_to_plot)
        plt.ylabel("Error (Actual-Predicted)")
        plt.title('Homoscedacity of Residuals');
        
    return lm, lm_pvalue, fvalue, f_pvalue



def independence(model, df, y, verbose, feature_to_plot):
    """Evaluate independence assumption of linear regression
    
    Parameters:
    -----------
    model : StatsModels model object based on df
    df : data used to create model
    verbose : optionally print metrics to console
    feature_to_plot : optionally display graph of residuals for selected feature
    
    Returns:
    --------
    vif_avg : average Variance Inflation Factor for all features
    
    """
    
    
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
        vif_avg = df_vif.VIF.mean()
        return vif_avg
    
    
    

    
def forward_selected(data, response):
    """Linear model designed by forward selection.

    SOURCE: https://planspace.org/20150423-forward_selection_with_statsmodels/
    
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
    """Apply recursive feature elimination (RFE) 

    Parameters:
    -----------
    x : DataFrame containing input features for RFE algorithm
    y : Series containing output variable used for RFE algorithm
    n : number of features to select and return
    
    Returns: 
    --------
    top_features : n features that results in a maximized R_squared using RFE
    
    """
    
    # Create linear regression object
    linreg = LinearRegression()
    
    # Apply RFE
    selector = RFE(linreg, n_features_to_select = n)
    selector = selector.fit(x, y.values.ravel())
    
    # Extract the top features
    top_features = x.loc[:, selector.support_]
    
    return top_features





def identify_continuous_features(df):
    """ Isolate columns with continuous features, excluding columns with few distinct values

    Parameters:
    -----------
    df : input DataFrame containing data to be analyzed
    
    Returns:
    --------
    df_continous_features : DataFrame with columns eliminated in cases of few distinct values
    
    """

    # Declare empty list for features
    continuous_features = []
    
    # Loop over all columns of input dataframe
    for col in df.columns:
        
        # Count the number of unique values in the column
        ct = df[col].value_counts().shape[0]
        
        # Subjective threshold to eliminate discrete features w/few unique values
        if ct > 50: 
            
            # Add to list of continuous features if relevant
            if (type(df[col][0])==type(np.int64(1))) or (type(df[col][0])==type(np.float64(0))):
                continuous_features.append(col)

    # Return dataframe containing only continuous features
    return df[continuous_features]