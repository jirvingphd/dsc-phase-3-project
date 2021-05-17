from project_functions.imports import *
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def plot_hist_regplot_gs(df,column,target='price',
                     figsize=(12,5),style='seaborn-notebook',
                     line_kws={'color':'black','ls':':'},
                    scatter_kws={'s':3},cat=False):
    
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize,constrained_layout=True)

        gs = fig.add_gridspec(nrows=2,ncols=3,)
        ax1 = fig.add_subplot(gs[0,2])
        ax2 = fig.add_subplot(gs[1,2])
        ax3 = fig.add_subplot(gs[:2,:2])
        

        if cat == True:
#             sns.barplot(data=df,x=column, y=target, ax=ax3,palette='dark',
#                         estimator=np.median)
            sns.stripplot(data=df,x=column,size=3, y=target,alpha=0.5, ax=ax3,palette='dark')
            hist_discrete = True
        else:
            # regplot
            hist_discrete = None
            sns.regplot(data=df,x=column, y=target, ax=ax3,
                        line_kws=line_kws, scatter_kws=scatter_kws)
        ## Histogram
        sns.histplot(data=df, x=column,stat='probability',discrete=hist_discrete,
                     ax=ax1)
                
        ## boxplot
        sns.boxplot(data=df,x=column,ax=ax2)
        
    fig.suptitle(f'Inspecting {column} vs {target}',y=1.05)
        
    return fig, (ax1,ax2,ax3)



def calc_vif(X_,drop=None,cutoff=5):
    """Calculates VIF scores for all columns.
    Modified from source: https://etav.github.io/python/vif_factor_python.html"""
    if drop is not None:
        X = X_.drop(columns=drop).copy()
    else:
        X = X_.copy()
    vif = pd.DataFrame()
    vif["features"] = X.columns
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['above_cutoff'] = vif['VIF Factor'] > cutoff
    return vif.round(2).sort_values('VIF Factor',ascending=False)

# calc_vif(df,drop=['price'])





def find_outliers_z(data):
    """Detects outliers using the Z-score>3 cutoff.
    Returns a boolean Series where True=outlier"""
    zFP = np.abs(stats.zscore(data))
    zFP = pd.Series(zFP, index=data.index)
    idx_outliers = zFP > 3
    return idx_outliers


def find_outliers_IQR(data):
    """Detects outliers using the 1.5*IQR thresholds.
    Returns a boolean Series where True=outlier"""
    res = data.describe()
    q1 = res['25%']
    q3 = res['75%']
    thresh = 1.5*(q3-q1)
    idx_outliers =(data < (q1-thresh)) | (data > (q3+thresh))
    return idx_outliers

def color_above_thresh(val,thresh=0.05):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val > thresh else 'black'
    return 'color: %s' % color


def check_null_cols(df,above_cutoff=True,percent_cutoff=5):
    """Displays null values for columns that have nulls > 0.
    Returns index of columns """
    ## Get Nulls, filter >0
    nulls = df.isna().sum()
    nulls = nulls[nulls>0]

    ## make into a df
    null_df = pd.DataFrame({'# null':nulls,
                            '% null': (nulls/len(df)*100).round(2)})
    
    null_df.sort_values('% null',ascending=False,inplace=True)
    null_df['Droppable'] = null_df['% null'] < percent_cutoff

    s = null_df
#     display(s.style.set_caption('Null Values '))
    return s

def column_report(df):
    """Returns a dataframe with the following summary information
    for each column in df.
    - Dtype
    - # Unique Entries
    - # Null Values
    - # Non-Null Values
    - % Null Values
    """
    report = pd.DataFrame({'dtype':df.dtypes, 
          
             'nunique':df.nunique(),
               '# Nulls': df.isna().sum(),
              '# Non-Nulls':df.notnull().sum(),
                          }).reset_index().rename({'index':'column'},axis=1)
    report[''] = range(len(report))
    report.set_index('',inplace=True)
    report['% null'] = np.round(report['# Nulls']/len(df)*100,2)
    return report



def make_ols(X,y,show_summary=True,diagnose=True):
    """Fits a statsmodels OLS on X and y. 
    Optionally displays the .summary and runs diagnose_model
    
    Returns: 
        model: fit statsmodels OLS"""
    
    model = sm.OLS(y,X).fit()
    
    ## Display summary
    if show_summary:
        display(model.summary())
        
    ## Plot Q-Qplot & model residuals
    if diagnose:
        try:
            fig,ax = diagnose_model(model,x_data=X)
            plt.show()
        except Exception as e:
            print('ERROR:')
            print(e)

    return model



def diagnose_model(model,x_data = None):
    """
    Plot Q-Q plot and model residuals from statsmodels ols model.
    
    Args:
        model (smf.ols model): statsmodels formula ols 
    
    Returns:
        fig, ax: matplotlib objects
    """
    
#     
    
    fig,ax = plt.subplots(ncols=2,figsize=(10,5))

    
    if x_data is None:
        xs = np.linspace(0,1,len(resids))
        resids = model.resid
    else: 
        y_hat = model.predict(x_data,transform=True)
        resids = y-y_hat
        
    sm.qqplot(resids, stats.distributions.norm,
              fit=True, line='45',ax=ax[0])        
    ax[1].scatter(x=y_hat,y=resids,s=2)
    ax[1].set(ylabel='Residuals',xlabel='Predicted')
    ax[1].axhline(0)
    ax[1].set_title('Residuals vs Preds')
    
    plt.tight_layout()
    plt.show()
    return fig,ax 