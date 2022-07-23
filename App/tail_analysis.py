import pandas as pd
import numpy as np

#%% drop dup function
def drop_dup(df):
    """ Drop duplicated indexs and row
    """
    ### Index deplicate ###
    uni_df = df.drop(df.index[df.index.duplicated()])
    ### row deplicate ###
    uni_df = uni_df.drop_duplicates()
    return uni_df

#%% pdf
def pdf(taget_se, bin_size):
    """ Get a pdf series of the time series
    args:
        taget_se (series): series of a single asset
        bin_size (inr/float): size of each group

    returns:
        pdf_se (series)
    """
    if bin_size <= 0:
        print ('The value of delta should be larger than 0')
        return
    min_edge = taget_se.min()
    max_edge = taget_se.max()
    if len(set([min_edge,max_edge])) == 1: # Cater for bott values are NaN values as well
        raise ValueError('The bin edges are the same, it means the target series only has 1 unique value')
    bin = np.linspace(min_edge, max_edge, bin_size)
    count = pd.value_counts(pd.cut(taget_se, bin)).sort_index()
    pdf_se = count/count.sum()
    pdf_se.index = [x.mid for x in pdf_se.index]
    return pdf_se
#%% cdf
def cdf(pdf_se):
    cdf_se = pdf_se.cumsum().iloc[::-1]
    return cdf_se
#%% right tail exceedence
def right_tail_exceedance(pdf_se):
    ex_se = pdf_se.iloc[::-1].cumsum()
    ex_se = ex_se.iloc[::-1]
    return ex_se
#%% left tail exceedence
def left_tail_exceedance(pdf_se):
    ex_se = pdf_se.cumsum()
    ex_se = ex_se.iloc[::-1]
    return ex_se
#%% remove inf/nan value
def remove_inf_nan(se):
    s = se.copy()
    s = s.replace([np.inf,-np.inf],np.nan).dropna()
    return s
#%% normalise exceedence
def normalised_exceedence(ex_se, p_target):
    '''Normalised the exceedence series (P(x)) with target p_value
    args:
        ex_se (series): exceedance series
        p_target (float): The target area for defining the tail
    returns:
        se (series): Normalised exceedence series
        cutoff_x (float): The position of x to get the p-value
    '''
    se = remove_inf_nan(ex_se)
    se = se / p_target # Normalise exceedances with P*
    if len(se[se == 1]) > 0:
        cutoff_x = se[se == 1].index[0]
    else:
        big_x = se[se < 1].index[0]
        small_x = se[se > 1].index[-1]
        big_y = se[se > 1].values[-1]
        small_y = se[se < 1].values[0]
        x_diff = np.abs(big_x) - np.abs(small_x)
        y_diff = np.abs(big_y) - np.abs(small_y)
        cutoff_x = small_x + (1-small_y) * (x_diff/y_diff)
    se.index = se.index / cutoff_x
    return se, cutoff_x
#%% create ln_ln_se
def ln_ln_se(nor_ex_se):
    """ 
    args:
        nor_ex_se (series): Normalised exceedence series
    returns:
        log log series, ln(Q(x)) & ln(x)
    """
    se = remove_inf_nan(nor_ex_se)
    se = -np.log(se) # cal Q = -ln(P/P*)
    se.index = np.log(se.index) # cal ln(u)
    return se