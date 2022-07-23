import streamlit as st
from table_of_content import Toc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import datetime
import tail_analysis as ta

st.set_page_config(page_title='StockTail',initial_sidebar_state='collapsed')
st.title("Live Stock Return Tail Analysis")
st.markdown("This dashboard can be used for different stocks tail analysis")

toc = Toc()
toc.placeholder(sidebar=True)

@st.cache(persist=True)
def load_stock_data(stock_ticker, start=None, end=None, interval='1d'):
    '''
    arg:
        stock_ticker (str): the ticker of stock
        start (str): date of the start of data e.g. '2018-01-01'
        end (str): date of the end of data e.g. '2020-01-01'
    return:
        data (df): stock data df
    '''
    data = yf.download(stock_ticker, start, end, interval)
    return data

### Retrieve data ###
toc.subheader("Stock Data Retrival")
ticker = st.text_input('Input data ticker: ', 'AAPL')
start_date = st.text_input("Start date of data: ", str(datetime.datetime.now().date()))
end_date = st.text_input("End date of data: ", str(datetime.datetime.now().date()))
interval = st.selectbox('Data interval (Dafult: 1 day): ',('1d', '1m', '1w', '1m'))
try:
    data = load_stock_data(ticker, start_date, end_date, interval)
except ValueError as ve:
    st.write(ve)

# Allow download data
start_date_file_name = start_date.replace('-','')
end_date_file_name = end_date.replace('-','')

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')
st.download_button('Download CSV', convert_df(data), f'{ticker}_{start_date_file_name}-{end_date_file_name}' '.csv', key='download-csv')

# Data display
st.dataframe(data, width=1000, height=250)

### Distribution Plot ###
toc.subheader("Distribution plot")
number_of_lag = st.slider("Number of lag: ", 1, 50)
target_series = st.selectbox('Target series: ',data.columns)

def get_lag_return(target_se, number_of_lag):
    return_data = target_se.pct_change(periods = number_of_lag, fill_method=None) # for (pt - pt-1)/pt-1
    return_data = return_data.dropna()
    return return_data

def get_lag_price_diff(target_se, number_of_lag):
    price_diff_data = target_se.diff(periods = number_of_lag) # for (pt - pt-1)
    price_diff_data = price_diff_data.dropna()
    return price_diff_data

def plot_distribution(series,number_of_lag, plot_title):
    series = series.dropna()
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.kdeplot(series, label = f'{ticker} {number_of_lag} lag', ax=ax)
    # plt.xlim([series.min(), series.max()])
    ax.legend(bbox_to_anchor=(1, 0.5))
    ax.set_title(plot_title)
    st.pyplot(fig)

return_series = get_lag_return(data[target_series], number_of_lag)
price_diff_series = get_lag_price_diff(data[target_series], number_of_lag)

if st.checkbox("Show return distribution plot", False):
    plot_distribution(return_series,number_of_lag, f'Return distribution from {start_date} to {end_date}')

if st.checkbox("Show price difference distribution plot", False):
    plot_distribution(price_diff_series,number_of_lag, f'Return distribution from {start_date} to {end_date}')

### Tail Analysis ###
toc.subheader("Tail Analysis")
tail_side = st.selectbox('Tail side: ',('LEFT', 'RIGHT'))
nb_bin = st.text_input("Number of bins for pdf: ", 50)
try:
    nb_bin = int(nb_bin)
except ValueError as ve:
    st.write('The input number of bins is not correct, will change to default value 50')
    nb_bin = 50

if st.checkbox("Show Ln-Ln plot", False):
### Tail Analysis - Plotting ###
    pdf_se = ta.pdf(return_series, nb_bin)
    if tail_side == 'LEFT':
        ex_se = ta.left_tail_exceedance(pdf_se)
    else:
        ex_se = ta.right_tail_exceedance(pdf_se)

    try:
        normalise_se, cutoff_x = ta.normalised_exceedence(ex_se, 0.05)
        ln_ln_se = ta.ln_ln_se(normalise_se[normalise_se.index>0])
        ln_ln_se = ta.remove_inf_nan(ln_ln_se)
        fig = plt.figure(figsize=(8, 5))
        ln_ln_se = ln_ln_se[ln_ln_se.index>0]
        plt.plot(ln_ln_se.index, ln_ln_se, 
                label=ticker)
        # plt.axvline(0)
        plt.title(f'{tail_side} tail normalised ln-ln plot for {ticker}')
        st.pyplot(fig)
    except IndexError:
        st.write('Please try to increase the number of bins, there is no tail part can be extracted')

### Tail Analysis - Slope calculation ###


toc.generate()