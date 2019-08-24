#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Model to Buy and Sell Stocks
# ### Specifically it uses a forest of 50 decision trees.  Each tree in our forest votes to either buy or sell a stock.  A position is only entered when 98% of our trees agree and a position is always held for 30 days then sold.  Results are promising but leaves room for improvements. Learning Period is 90 days.  
# #### By Mike Clark 8/9/2019
# 

# In[30]:


#https://arxiv.org/pdf/1605.00003.pdf
#http://ahmedas91.github.io/blog/2016/07/01/predicting-us-equities-trends-using-random-forests/


import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
from talib import abstract as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.plotly
import cufflinks as cf

cf.go_offline()
init_notebook_mode()


# ### This function returns a bunch of stock indicators (features) for each stock which our machine learning program will be using to predict whether to buy our sell.  We need to give our model features that will be useful for predicting future stock price.

# In[31]:


def get_indicators(stockData, period):
    stocks_indicators = {}
    
    #data = pd.DataFrame(SMA(stockData, timeperiod=5))
    #data.columns = ['sma_5']
    #data['sma_10'] = pd.DataFrame(SMA(stockData, timeperiod=10))
   
    #features['diff_sma'] = data['sma_10'] - data['sma_5']
   
    features =  pd.DataFrame()
    bands = ta.BBANDS(stockData, timeperiod=8, nbdevup=2, nbdevdn=2)
    
    features['BB'] = (stockData['close'] - bands['middleband'])/((bands['upperband']-bands['lowerband'])/2)
    features['close'] = stockData['close']
    #features = pd.concat([features,STOCHF(stockData,  
    #                                 fastk_period=14, 
    #                                   fastd_period=3)],
    #                      axis=1)
    #features['MA50day']= pd.DataFrame(SMA(stockData, timeperiod=50))
    features['NATR'] = ta.NATR(stockData,timeperiod=14)
    #features['adx'] =ADX(stockData,timeperiod=14) Not Good
    #features['macd'] = pd.DataFrame(MACD(stockData, fastperiod=12, slowperiod=26)['macd'])
    features['rsi'] = pd.DataFrame(ta.RSI(stockData, timeperiod=14))
    features['profit'] = ta.ROC(stockData, timeperiod=period)
    features['profit'] = features['profit'].shift(-period)
    features['pct_change'] = ta.ROC(stockData, timeperiod=period)
    features['pct_change'] = features['pct_change'].shift(-period)
    features['pct_change'] = features['pct_change'].apply(lambda x: 1 if x > 0 else -1 if x <= 0 else np.nan)
    features = features.dropna()
        
    #=============Not currenctly using these features below =====================
    #features[['diff_sma', 'sma_10']].sub(features['sma_5'], axis=0)
    #features['mom_10'] = pd.DataFrame(MOM(stockData, 10))
    
    #data['wma_10'] = pd.DataFrame(WMA(stockData, 10))
    #data['wma_5'] = pd.DataFrame(WMA(stockData,5))
    #features['diff_wma'] = data['wma_10'] - data['wma_5']

        
    #features = pd.concat([features,STOCHF(stockData,  
    #                                fastk_period=14, 
    #                                 fastd_period=3)],
    #                   axis=1)
    #features['willr'] = pd.DataFrame(WILLR(stockData, timeperiod=14))
    #features['cci'] = pd.DataFrame(CCI(stockData, timeperiod=14)) 
    #features['adosc'] = pd.DataFrame(ADOSC(stockData, fastperiod=3, slowperiod=10)) 
                                      
    
    return features


# # Code to create buy signals using historical stock data

# In[32]:


# number of days of stock data we will be using in our window of time.
points=90
start_delta=points
# Number of days we will be holding our stock
period = 30
# Number of decision trees in our forest
trees = 50
# Funny Money
balance = 100000
# Our trading cost of buy and then selling a stock.
tradeCost= 14
# Each of our positions will be $10000 bucks worth of stock
position=10000

# List of stock symbols
#top_500 = ['MSFT', 'XOM',  'GE', 'T', 'VZ']
#top_500 = ['SPY','BAC','BLL','AMGN','AES','AMD','MMM','MSFT', 'GE', 'T', 'VZ', 'AMZN']
#stockSymbols=['BAC','BLL','AMGN','AES','AMD','MMM','MSFT', 'GE', 'T', 'VZ', 'AMZN',
#             'BBT', 'HRB', 'BXP','IVV','EFA','AGG','IEFA','IEMG','IJH','IWM','IJR',
#             'LQD','EEM','SHY','USMV']
#stockSymbols = ['BAC','BLL','AMGN','AES','AMD','MMM','MSFT', 'GE', 'T', 'VZ', 'AMZN']
#stockSymbols=['MSFT','AMD']
stockSymbols=['AMD']
# with energry
#top_500 = ['SPY','BAC','BLL','AMGN','AES','AMD','MMM','MSFT', 'XOM',  'GE', 'T', 'VZ', 'AMZN', 'XLE']
#top_500 = ['SPY']


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()


# Let's go out to the web and read some historical data for our list of stocks
stocks = {}
for i in stockSymbols:
    stocks[i] = web.get_data_yahoo(i,start,end)


for i,j in enumerate(stocks):
    stocks[j].columns = [s.lower() for s in stocks[j].columns]
    stocks[j].volume = stocks[j].volume.apply(lambda x: float(x))
    stocks[j].close = stocks[j].close.apply(lambda x: float(x))
    


results = pd.DataFrame(columns=['Symbol','Trade Type', 'Profit', '% return per trade'])
buySignals = pd.DataFrame(columns=['Symbol'])
#results.set_index('Date')
    
for symbol in stockSymbols:
    print("processing stock: ", symbol)
    for inc in range(2*points, stocks[symbol]['open'].count()):
       
        stockData = stocks[symbol].iloc[:inc]
        features = get_indicators(stockData,period)
        
        #to be relistic we will not use the datapoints in the period for training.  The period is also
        # the number of days we will be holding our stock before selling.
        train = features.iloc[-(points+period):-period,:]
        x_train = train.iloc[:,:-2]  # don't want profit column or return in our training set
        y_train = train.iloc[:,-1]  # this is the return after holding our stock for the holding period
        
        

        #print x_train, y_train

        rf_model = RandomForestClassifier(trees)
        
        #for i in range(5):
        rf_model.fit(x_train,y_train)
        try:
            #take only the last date of data.  This will be the features we have the day we our deciding 
            # enter into a trade position
            X = features.iloc[-1,:-2].values.reshape(1, -1) 
            predict_value = rf_model.predict(X)
            #Sort of probability of the prediction being correct, prob_Sell + prob_Buy always add up to 1.  
            # It's more  correct to say they they are the ratio of votes by the trees of our forest,
            # eg. If prob_Sell is 0.2 then 20% of the trees are voting to sell the stock.
            prob_Sell,prob_Buy = rf_model.predict_proba(X)[0]
        except:
            print("Going To Next Stock Symbol: Not enough data for prediction")
            features.to_csv('errors.csv')
            #x_train.to_csv('x_train')
            #y_train.to_csv('y_train')
            continue
            #raise
            
        
        
        
        # Logic to go long with a stock, betting it will go up in price
        if(prob_Buy>0.98):
            tmp_signals = pd.DataFrame({'Symbol':symbol}, index=[pd.to_datetime(features.index[-1])])
            buySignals = buySignals.append(tmp_signals)


buySignals.index.names=['Date']


print("====================  Done !!!! =========================")


# In[ ]:


buySignals.to_csv('buySignals_test.csv')

