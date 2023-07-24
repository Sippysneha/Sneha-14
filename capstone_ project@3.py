#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT- DATA VISULAIZATION\ Exploratory Data Analysis
# 
# # Data Collection and Data Cleaning
# 

# # importing all required Libraries #

# In[118]:



import pandas as pd
import numpy as np
import seaborn as  sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import plotly
import plotly.graph_objs as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
plotly.offline.init_notebook_mode (connected = True)
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib as plt


# # Load the Dataset #

# In[119]:


df = pd.read_csv(r'D:\Bia projects\capstone project\Stock Market Snehal sippy\OHLC.csv')


# In[120]:


df


# # view of Dataset #

# In[121]:


df.head(10)


# In[122]:


df.tail(10)


# # Checking numerical and categorical data  #
# 

# In[123]:


df.dtypes


# Data type of 'Date' , 'Symbol'& 'Vol' is object. Rest of the data is in numerical form which is appropriate
# 
# 

# In[124]:


# checking any Duplicates data


# In[125]:


df[df.duplicated()]


# # shape of the dataset
# 

# In[126]:


df.shape


# # some statistical information about data

# In[127]:


df.describe()


# #  Checking NULL values
# 

# In[128]:


df.isna().sum()


# In[129]:


df[df.isnull().any(axis = 1)]


# As there are very less NaN rows so we will drop it.

# In[130]:


#Dropping rows with null values
df.dropna(axis = 0,inplace = True)


# In[131]:


#again checking any NaN values
df.isnull().sum()


# In[132]:


df


# # Formatting data(Making datatypes compatible)

# In our dataset only 'Date' & 'Vol' are in object form which need to be changed to 'DateTime' format and in integer format.
# 
# 90

# In[133]:


df.dtypes


# #Converting Date column to 'Date Time'

# In[134]:


df['Date'] = pd.to_datetime(df['Date'].str.split('.').str[0], format='%Y-%m-%d')


# In[135]:


df


# In[136]:


df.Symbol.nunique()


# #Converting the 'Vol' column to integer form.
# 

# In[137]:


df['Vol'] = df['Vol'].str.replace('(', '').str.replace(')', '').str.replace(',', '')


# In[138]:


df['Vol'] = df['Vol'].astype(int)


# In[139]:


df['Vol']


# In[140]:


#Again check the datatypes
df.dtypes


# In[141]:


#sort the data according to 'Date'
df.sort_values('Date',inplace = True)


# In[142]:


df.head(10)


# In[143]:


df


# # Setting Date column as index

# In[144]:


df.set_index(df['Date'],drop=True,inplace=True)


# In[145]:


df


# # total no of companies listed

# In[146]:


df.Symbol.nunique()


# # calculating the Turnover of all companies

# In[147]:


df['Turnover'] = df['Close'] * df['Vol']


# In[148]:


df


# # Sorting companies with HIGH Turnover

# In[149]:


sorted_data = df.sort_values('Turnover', ascending=False)


# In[150]:


sorted_data


# In[151]:


df1=sorted_data.sort_values(by="Turnover",ascending=False)[:20].reset_index(drop=True)
df1


# In[154]:


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# In[155]:


color_palette = sb.color_palette("plasma")  
plt.figure(figsize=(10, 6))
bars = plt.bar(df1['Symbol'], df1['Turnover'],color = color_palette)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')
plt.xlabel('Symbol')
plt.ylabel('Turnover')
plt.title('Top 20 Companies with High Turnover')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[156]:


fig=px.scatter(df1,x='Symbol',
              y='Turnover',
               color='Turnover',
               size='Turnover'
              )
fig.update_layout(title="Top 20 Highest Turnover Symbols",
                 title_x=.5)

fig.show() 


# In[157]:


top_5_companies = sorted_data.head(5)


# In[158]:


top_5_companies


# In[159]:


color_palette = sb.color_palette("viridis")  
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5_companies['Symbol'], top_5_companies['Turnover'],color = color_palette)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')
plt.xlabel('Symbol')
plt.ylabel('Turnover')
plt.title('Top 5 Companies with High Turnover')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# #  companies with LOW Turnover

# #checking companies with Zero Turnover

# In[160]:


df


# # checking & removing companies with Zero Turnover

# In[161]:


df = df[df['Turnover'] != 0]
df


# # Sorting companies with Low turnover

# In[162]:


df1=df.sort_values(by="Turnover",ascending=True)[:20].reset_index(drop=True)
df1


# In[163]:


color_palette = sb.color_palette("plasma")  
plt.figure(figsize=(10, 6))
bars = plt.bar(df1['Symbol'], df1['Turnover'],color = color_palette)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')
plt.xlabel('Symbol')
plt.ylabel('Turnover')
plt.title('20 Companies with low Turnover')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[164]:


fig=px.scatter(df1,x='Symbol',
              y='Turnover',
               color='Turnover',
               size='Turnover'
              )
fig.update_layout(title="Top 20 Lowest Turnover Symbols",
                 title_x=.5)

fig.show()


# In[165]:


low_5_companies = df1.head(5)


# In[166]:


low_5_companies


# In[167]:


color_palette = sb.color_palette("viridis")  
plt.figure(figsize=(10, 6))
bars = plt.bar(low_5_companies['Symbol'], low_5_companies['Turnover'],color = color_palette)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')
plt.xlabel('Symbol')
plt.ylabel('Turnover')
plt.title('5 Companies with low Turnover')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#  # Univariate analysis
# 

# # Data analysis using Data Visualization
# 

# In[168]:


df


# In[169]:


df.drop(['S.no'],axis=1,inplace=True)


# In[170]:


df


# In[171]:


temp_df=df[['Open','High','Low','Close','Symbol','Date']]
temp_df=temp_df[temp_df['Date']<"2013"]
temp_df=temp_df[temp_df['Symbol']=="EBL"]
dates=temp_df['Date']
temp_df=temp_df[temp_df.columns[:-1]]
temp_df.head()


# In[172]:


temp_df.plot(figsize=(15,9))
pl.grid(alpha=.5)
pl.title("OLHC of EBL")
pl.show()


# In[173]:


fig=px.scatter(temp_df,x='High',
              y='Low',
               color='High',
              size='Low',
              )
fig.update_layout(title=" Scatter plot ratio of High and Low",
                 title_x=.5)
fig.show()


# # Finding the highest Turnover company

# In[174]:


df['Turnover'].max()


# In[175]:


idx_max_turnover = df['Turnover'].idxmax()

# Retrieve the data of the company with the highest turnover
company_with_high_turnover = df.loc[idx_max_turnover]

print(company_with_high_turnover)


# In[176]:


df2=company_with_high_turnover.sort_values(by="Turnover",ascending=False)[:20].reset_index(drop=True)
df2


# In[177]:


df2.iloc[0]


# # 'EBLPO recorded with highest Turnover of Rs 1917639392 in 2019-02-14 00:00:00'.

# # MOST Sold Stocks

# In[178]:


df


# In[179]:


temp_df=df[['Symbol','Vol','Date']].sort_values(by='Vol',ascending=False)[:20]
temp_df


# # Let'S plot the companies with high Volume

# In[180]:


fig=go.Figure(
    go.Bar(
    x=temp_df.Symbol,
    y=temp_df.Vol,
    textposition="inside",
    marker=dict(color=temp_df.Vol,
                   colorscale='balance'),
    text=temp_df.Vol,
    )
    )

fig.update_layout(title="Top 20 Highest Recorded Volumes/ Sales",
                 title_x=.5,
                 barmode='stack', xaxis={'categoryorder':'total descending'}
                 )

fig.show()


# In[181]:


df.Symbol.nunique()


# # Data with close price less than 50

# In[182]:


df[df['Close']<50].head(20)


# In[183]:


df['Close'].nunique()


# # Data with Close price more than 5 thousand

# In[184]:


Close_high=df[df['Close'] > 5000].set_index("Date")


# In[185]:


Close_high


# In[186]:


companies=Close_high['Symbol'].unique()
companies


# 'UNL', 'NLIC', 'RBS', 'RBCL', 'RBCLPO', 'CIT', 'BNT', 'NBBL' are Companies having High CLosing Price.

# In[187]:


df['Close'].sort_values(ascending=False)


# In[188]:


df[df['Close'] == 36000].set_index("Date")


# # Unilever company with symbol 'UNL' has highest Open and highest Close price

# In[189]:


unilever_data=df[df['Symbol']=='UNL'].set_index('Date')
unilever_data


# In[190]:


unilever_data.head()


# Let's plot OHLC of 'UNL'

# In[292]:


unilever_data[['Open','High','Low','Close']].plot(figsize=(15,10))
plt.grid()
plt.title("UNL Data")
plt.show()


# In[192]:


RBCL_data=df[df['Symbol']=='RBCL'].set_index('Date')
RBCL_data


# """Let's plot OHLC of RBCL"""

# In[293]:


RBCL_data[['Open','High','Low','Close']].plot(figsize=(15,10))
plt.grid()
plt.title("RBCL Data")
plt.show()


# In[194]:


RBCLPO_data=df[df['Symbol']=='RBCLPO'].set_index('Date')
RBCLPO_data


# #Let's plot the OHLC of 'RBCLPO'

# In[294]:


RBCLPO_data[['Open','High','Low','Close']].plot(figsize=(15,10))
plt.grid()
plt.title("RBCLPO Data")
plt.show()


# # Companies with Closing price more than 15,000

# In[196]:


Close_high=df[df['Close'] > 15000].set_index("Date")
Close_high


# In[197]:


companies=Close_high['Symbol'].unique()
companies


# # 'UNL','RBCLPO','RBCL' companies having highest CLOSING PRICE

# In[200]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Close"])
    axs.grid()
    axs.set_title(f"Closing price of {comp}")


# In[201]:


Open_high=df[df['Open'] > 15000].set_index("Date")
Open_high


# In[202]:


companies=Open_high['Symbol'].unique()
companies


# In[203]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Open_high[Open_high["Symbol"]==comp]["Open"])
    axs.grid()
    axs.set_title(f"Opening price of {comp}")


# In[206]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["High"])
    axs.grid()
    axs.set_title(f"High price of {comp}")


# In[207]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Low"])
    axs.grid()
    axs.set_title(f"Low price of {comp}")


# In[235]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Close"],label="Close")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Open"],label="Open")
    axs.grid()
    axs.legend()
    axs.set_title(f"Open/Close price of {comp}")


# # Plotting the OHLC OF all 3 companies

# In[210]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Close"],label="Close")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Open"],label="Open")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Low"],label="Low")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["High"],label="High")
    axs.grid()
    axs.legend()
    axs.set_title(f"OHLC price of {comp}")


# # Plotting the Volume all 3 Companies

# In[212]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["Vol"],label="Volume")
    axs.grid()
    axs.legend()
    axs.set_title(f"Sales Volume price of {comp}")


# # FINDING the rolling mean of all 3 Companies.

# In[213]:


ma_days = [10, 20, 50]
for ma in ma_days:
    for company in companies:
        column_name = f"MA for {ma} days"
        Close_high[column_name] =Close_high['Close'].rolling(ma).mean()


# In[215]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["MA for 10 days"],label="MA for 10 days")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["MA for 20 days"],label="MA for 20 day")
    axs.plot(Close_high[Close_high["Symbol"]==comp]["MA for 50 days"],label="MA for 50 days")    
    axs.grid()
    axs.legend()
    axs.set_title(f"Sales Volume of {comp}")


# # Finding the percentage of change in close price than previous close

# In[216]:


for company in companies:
    Close_high['D_Return'] = Close_high['Close'].pct_change()
    


# In[217]:


Close_high


# In[219]:


fig,axes=plt.subplots(3,figsize=(10,14))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(Close_high[Close_high["Symbol"]==comp]["D_Return"],label="Daily Returns")    
    axs.grid()
    axs.legend()
    axs.set_title(f"Daily Return  of {comp}")

 


# In[220]:


Close_high


# In[261]:


UNL_data = Close_high[Close_high['Symbol'] == 'UNL']


# In[295]:


UNL_data


# In[296]:


UNL_data.sort_values(by='Date',ascending=True,inplace=True)
UNL_data


# In[297]:


UNL_data.head()


# In[298]:


UNL_data.tail()


# In[299]:


UNL_data.describe()


# In[300]:


UNL_data.describe().style.background_gradient(cmap="Set3")


# In[301]:


UNL_data.isnull().sum()


# In[302]:


UNL_data


# In[303]:


import matplotlib.pyplot as plt


# In[304]:


UNL_data


# # Checking stationarity

# In[305]:


UNL_data


# #First, we need to check if a series is stationary or not because time series analysis only works with stationary data.

# In[306]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(Close):
    result=adfuller(Close)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ") 
    


# In[307]:


adfuller_test(df['Close'])


# # AS APPLYING Adafuller test on closing price it is obsereved that data is stationary.
# so we can apply ARIMA model on the data to predict the close price.

# In[308]:


df


# In[356]:


df_unl=df[['Symbol','Date','Open','High','Low','Close','Vol','Turnover']]
df_unl=df_unl[df_unl['Date']>"2015"]
df_unl=df_unl[df_unl['Symbol']=="UNL"]
dates=df_unl['Date']
df_unl=df_unl[df_unl.columns[:-1]]
df_unl.head()


# In[357]:


df_unl


# In[358]:


df_unl.isnull().sum()


# In[359]:


df_unl.describe()


# In[360]:


closing_prices = df_unl['Close']

# Create a plot for the closing price
plt.figure(figsize=(10, 6))
plt.plot(closing_prices.index, closing_prices, marker='o', linestyle='-')

# Set plot title and labels
plt.title('Closing Price of the Stock')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# In[361]:


for company in companies:
    df_unl['D_Return'] = df_unl['Close'].pct_change()


# In[362]:


df_unl


# In[363]:


df_unl['D_Return'].max()


# In[364]:


fig,axes=plt.subplots(2,figsize=(6,10))
for comp,axs in zip(companies,axes.ravel()):
    axs.plot(df_unl[df_unl["Symbol"]== "UNL"]["D_Return"],label="Daily Returns")    
    axs.grid()
    axs.legend()
    axs.set_title("Daily Return  of UNL")


# FROM the above obervastion it is observed that % of daily return on price is more than 9% in year 2015-2016.
# Again in year 2019-2020 it reached 9.5% .

#  Checking stationarity
#  

# In[365]:


closing_prices = df_unl['Close']

# Perform the Augmented Dickey-Fuller test
result = adfuller(closing_prices)

# Extract the test statistics and p-value
test_statistic = result[0]
p_value = result[1]

# Print the results
print(f'Test Statistic: {test_statistic}')
print(f'P-Value: {p_value}')

# Interpret the results
if p_value <= 0.05:
    print("P-value is less than or equal to 0.05. The data is stationary.")
else:
    print("P-value is greater than 0.05. The data is non-stationary.")


# AS it is observed that using ADF test the data of UNL is non-stationary. AS ARIMA model of TIME SERIES Works on Stationary data.
# so let's make data stationary using differencing.

# # making Series Stationary

# In[368]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


# In[371]:


# Perform differencing on the 'Close' column
df_unl['Close_diff'] = df_unl['Close'].diff()

# Plot the original 'Close' and differenced 'Close_diff' columns
plt.figure(figsize=(12, 6))
plt.plot(df_unl['Close'], label='Original Close')
plt.plot(df_unl['Close_diff'], label='Differenced Close')
plt.legend()
plt.title('Original Close vs. Differenced Close')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[372]:


#Determing rolling statistics

rolling_window_size = 30
df_unl['Rolling_Mean'] = df_unl['Close_diff'].rolling(window=rolling_window_size).mean()
df_unl['Rolling_Std'] = df_unl['Close_diff'].rolling(window=rolling_window_size).std()

# Plot the original 'Close_diff' column, rolling mean, and rolling standard deviation
plt.figure(figsize=(12, 6))
plt.plot(df_unl['Close_diff'], label='Differenced Close')
plt.plot(df_unl['Rolling_Mean'], label=f'Rolling Mean (Window={rolling_window_size})', color='red')
plt.plot(df_unl['Rolling_Std'], label=f'Rolling Std (Window={rolling_window_size})', color='green')
plt.legend()
plt.title('Differenced Close with Rolling Mean and Rolling Std')
plt.xlabel('Date')
plt.ylabel('Differenced Close')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[376]:


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
lag_acf = acf(df_unl['Close_diff'], nlags=20)
lag_pacf = pacf(df_unl['Close_diff'], nlags=20)


# In[377]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_unl['Close_diff'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_unl['Close_diff'],lags=40,ax=ax2)


# In[375]:


# Plot ACF (Autocorrelation Function)
plt.figure(figsize=(12, 6))
plot_acf(df_unl['Close_diff'], lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.xticks(range(0, 41, 5))
plt.grid(True)
plt.show()

# Plot PACF (Partial Autocorrelation Function)
plt.figure(figsize=(12, 6))
plot_pacf(df_unl['Close_diff'], lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.xticks(range(0, 41, 5))
plt.grid(True)
plt.show()


# In[386]:


from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


# In[387]:



type(df_unl['Close_diff'])


# In[401]:


actuals=df_unl['Close_diff'].tail(30)


# In[402]:


actuals


# In[403]:


order = (1, 1, 1)  # Example order for ARIMA(1, 1, 1) model

model = sm.tsa.ARIMA(df_unl['Close_diff'], order=order)
results = model.fit()

# Print the model summary
print(results.summary())

# Get the predicted values (forecast)
forecast_values = results.forecast(steps=30)  # Adjust the number of steps as needed for forecasting

# Plot the original 'Close_diff' data and the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df_unl['Close_diff'], label='Original Close_diff')
plt.plot(forecast_values, label='Forecasted Close_diff', color='red')
plt.legend()
plt.title('Original Close_diff vs. Forecasted Close_diff')
plt.xlabel('Date')
plt.ylabel('Differenced Close')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[405]:


actuals = df_unl['Close_diff'].tail(30)


# In[406]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(actuals, forecast_values)
print('MAE: %f' % mae)


# In[ ]:




