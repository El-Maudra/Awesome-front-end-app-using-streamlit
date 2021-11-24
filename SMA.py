import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
import warnings
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Trading Strategy Based on Simple Moving Averages')

st.write('Loading the data')

spy = pd.read_csv('SPY.csv', index_col='Date', parse_dates=True)

st.markdown('#### SPY Dataset')
st.write(spy.head())

st.write('Let us create a new data with the Close column and name it price and create simple moving averages for the first 42 days and 252 days.')
df_spy = spy[['Close']].rename(columns={'Close':'Price'})
df_spy['SMA1'] = df_spy['Price'].rolling(42).mean()
df_spy['SMA2'] = df_spy['Price'].rolling(252).mean()
st.write(df_spy.sample(5))

st.markdown('#### Visualization of the data')
def viz(df):
    plt.figure(figsize=(20, 10))
    plt.title('S&P 500 Historical Closing Price | 42 & 252 days SMAs', fontsize=28)
    plt.plot(df[df.columns[0]], label=f'{df.columns[0]}')
    plt.plot(df[df.columns[1]], label=f'{df.columns[1]}')
    plt.plot(df[df.columns[2]], label=f'{df.columns[2]}')
    plt.legend()
    
fig = viz(df_spy)
st.pyplot(fig)

    
st.markdown('Remember the rule is to go long whenever the shorter SMA is above the longer one and vice-versa. This is because you want to short when the prices are high and long when the prices are low. The ideology behind this is that when the prices are high, you want to sell so that you get a profit. And when the prices are low, you want to hold your position (longing) so that when the markets bounces back, you will sell for profit (shorting)')

st.markdown('#### Visualizing the position over time')
st.write('1 representes an expression SMA > SMA2;')
st.write('-1 represents SMA2 < SMA1')
df_spy['positon'] = np.where(df_spy['SMA1'] > df_spy['SMA2'], 1, -1)

df_spy.dropna(inplace=True)
st.write(df_spy.sample(5))

st.write('S&P 500 Market Positioning')
fig_, ax_ = plt.subplots()
ax_ = plt.plot(df_spy['positon'])
plt.xlabel('Position')
plt.ylabel('Market Performance')
st.pyplot(fig_)

st.markdown('### Calculating the performance of the strategy')
st.write('To do this, we calculate the log returns based on the original financial time series.')
df_spy['Returns'] = np.log(df_spy['Price'] / df_spy['Price'].shift(1))

st.write('Frequency distribution of S&P 500 log returns')
fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(df_spy['Returns'])
plt.xlabel('Return')
plt.ylabel('Density')
st.pyplot(fig_mpl)

st.write('To derive the strategy returns, multiply the positions column --shifted by one trading day-- with the returns column. Since log returns are additive, calculating the sum over the columns returns and strategy provides a first comparison of the performance of the strategy relative to the base investment itself.')
df_spy['strategy'] = df_spy['positon'].shift(1) * df_spy['Returns']
st.text('For illustration only;\n')
st.write(df_spy[['Returns', 'strategy']].sum())

st.text('For gross performance;\n')
st.write(df_spy[['Returns', 'strategy']].sum().apply(np.exp))

st.subheader('Calculating the average, annualized risk statistics for both the stock and the strategy')
st.write('Use the sidebar(Top Left) to find the statistic')
Average = df_spy[['Returns', 'strategy']].mean() * 252
Annual_Average = np.exp(df_spy[['Returns', 'strategy']].mean() * 252) -1
Risk = df_spy[['Returns', 'strategy']].std() * 252 ** 0.5
Annual_Risk = (df_spy[['Returns', 'strategy']].apply(np.exp)-1).std() * 252 ** 0.5
stats = [Average, Annual_Average, Risk, Annual_Risk]
names = {'Average':stats[0], 
         'Annual_Average':stats[1], 
         'Risk':stats[2], 
         'Annual_Risk':stats[3]}
dropdown = st.sidebar.multiselect('Select the Risk statistic', names)

for j in dropdown:
    st.write(j)
    st.write(names.get(j))

st.subheader('Calculating the Maximum Drawdown and Longest Drawdown periods')
df_spy['cumret'] = df_spy['Returns'].cumsum().apply(np.exp)
df_spy['cummax'] = df_spy['cumret'].cummax()
st.write('If strategy performed better, put strategy instead of returns')
st.write(df_spy.sample(5))

def drawdown(df):
    plt.figure(figsize=(20, 10))
    plt.title('Gross Perfomance and Cumulative Maximum Perfomance of the SMA based strategy', fontsize=28)
    plt.plot(df['cumret'], linewidth=2.5, label='cumret')
    plt.plot(df['cummax'], linewidth=2.5, label='cummax', color='green')
    plt.xlabel('Date', fontsize=20)
    plt.legend(fontsize=15)
    
fig2 = drawdown(df_spy)
st.pyplot(fig2)

st.markdown("The maximum drawdown is then calculated as the maximum of the difference between the two relevant column;")
st.markdown("**cummax - cumret**")
st.markdown("Longest drawdown requires the dates at which the gross performance equals its cummulative maximum i.e where new maximum is set.")

drawdown = df_spy['cummax'] - df_spy['cumret']
#print(f'The Maximum Drawdown is: {drawdown.max()}')
temp = drawdown[drawdown==0]
periods = (temp.index[1:].to_pydatetime() - 
           temp.index[:-1].to_pydatetime())
#print(f'The Maximum drawdown: {drawdown.max()} \nThe Longest drawdown period: {periods.max()}')

drawdown_period = {'Maximum Drawdown':drawdown.max(), 
                   'Longest Drawdown':periods.max()}

bar = st.multiselect("Select the Drawdown Period", drawdown_period)
for x in bar:
    st.write(x)
    st.write(drawdown_period.get(x))
    
st.subheader('Conclusions')
st.write("The stock used above is SPY(S&P 500). I would recommend you to tryout different financial instruments i.e bitcoins or currency pairs such as EUR/USD! Using SPY as a stock could be misleading since you may treat it as a single stock while it just tracks the perfomance of Fortune 500 company stocks as a whole. Thus you may look at it as the historical market perfomance.")

st.write("From the two SMAs, it looks like we are going to hold a short position of the stock. The Gross perfomance of S&P 500 out perfomance the Simple Moving Average Based Strategy since we can see that the stock the passive investment benchmark is better. Our stock has a gaussian distribution. From the stock we can see that the Maximum drawdown is 0.32 and the Longest drawdown is 418 days. Pretty long huh!")
st.write('You have reached the end of this app! To find the source code of this analysis')







