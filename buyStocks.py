# SOME CLASSES TO MANAGE MONEY
class Balance:
    balance = 100000.00
    balanceCheckPoints = {}
    positionsWorth = {}
    netWorth = {}
    
    
    def setCheckPoint(self, date, stockData):
        Balance.balanceCheckPoints[date] = Balance.balance
        positionsWorth = Broker.positionsWorth(date,stockData)
        Balance.positionsWorth[date] = positionsWorth
        Balance.netWorth[date] = Balance.balance + positionsWorth
        

    def subtract(self, amount):
        Balance.balance -= amount
    

    def add(self, amount):
        Balance.balance += amount

# SOME CLASSES TO BUY, SELL AND PRINT REPORT

import math


class OpenPosition:
    def __init__(self, symbol,date,shares, data):
        self.stockData = stockData
        self.symbol = symbol
        self.shares = shares
        self.buyPrice = data.close
        self.buyDate = pd.to_datetime(date)
        self.buyWorth = shares * self.buyPrice
        self.sellDate = data.sellDate
        
    def worth(self,data):
        worth = self.shares * data.close
        return worth
    
class ClosedPosition:
    def __init__(self, openPosition, dateClosed, data):
        self.positionTarget = 100000
        tradeCost = 14
        
        self.symbol = openPosition.symbol
        self.shares = openPosition.shares
        self.buyPrice = openPosition.buyPrice
        self.soldPrice = data.close
        self.buyDate = openPosition.buyDate
        self.soldDate = pd.to_datetime(dateClosed)
        
        self.sellWorth = self.shares * data.open
        self.profit = self.shares * (self.soldPrice - self.buyPrice) - tradeCost
        self.percentProfit = 100.00 * (self.profit/self.sellWorth)
        
class Plan:
    positionTarget = 10000
    def checkBuy(self, symbol, date, data):
        if(symbol in positions):
            return False
        elif(balance.balance < self.positionTarget):
            return False
        else:
            return True
        
    def checkSell(self, position, date, data):
        #print(pd.to_datetime(date), position.buyDate)
        #print((pd.to_datetime(date)- position.buyDate).days)
        #if((pd.to_datetime(date)- position.buyDate).days >= 30):
        if(date>=position.sellDate):
            return True

class Broker:
    def __init__(self, plan):
        self.plan =  plan
    
    @staticmethod
    def positionsWorth(date,stockData):
        worth = 0
        for symbol,position in positions.items():
            data = stockData[position.symbol].loc[date]
            worth+=position.worth(data)
        return worth
            
        
    def checkBuys(self, symbol, date, data):
        # check to see if we should buy the position
        if(self.plan.checkBuy(symbol, date, data )):
            shares = math.floor(self.plan.positionTarget/data.open)
            openPosition = OpenPosition(symbol,date,shares, data)
            positions[symbol] = openPosition
            balance.subtract(openPosition.buyWorth)
            # set balance check point
            #Balance.setCheckPoint(date, data)
    
    def checkSells(self, position, symbol, date, data):
            if(self.plan.checkSell(position,date,data)):
                closedPosition = ClosedPosition(position, date, data)
                closedPositions.append(closedPosition)
                balance.add(closedPosition.sellWorth)
                soldPositions.append(symbol)
                #Balance.setCheckPoint(date, data)
                
                
class Report:
    
    def printPerform(self):
        
        profit = 0
        profitable = 0
        
        
        for closedPosition in closedPositions:
            print( closedPosition.symbol,"\t",
                  closedPosition.buyDate.strftime("%Y-%m-%d"),"\t" ,
                  closedPosition.soldDate.strftime("%Y-%m-%d"),"\t" ,
                  '{:,.2f} %'.format(closedPosition.percentProfit))
                  
            if(closedPosition.profit > 0):
                profitable+=1
                
            profit += closedPosition.profit
            



        pctProfitable = 100 * profitable/len(closedPositions)
                
            
        
        print("*******************************")   
        print("profit= \t\t{:,.2f}".format(profit))
        print("percent profitable= \t{:,.2f}".format(pctProfitable))


### Returns a dataframe of trade results
def getResults():
## Lets create a dataframe to evaluate results using pandas
    
    results = pd.DataFrame(columns=['Buy Date','Symbol','Trade Type',  'Profit', '% return per trade',
                                'Shares','Buy Price','Sold Price'])

 
    
    for closedPosition in closedPositions:
        tmp = pd.DataFrame({'Buy Date':closedPosition.buyDate.strftime("%Y-%m-%d"),'Symbol':closedPosition.symbol,
                            'Trade Type':'Long','Profit':closedPosition.profit, 
                            '% return per trade':closedPosition.percentProfit,
                            'Shares':closedPosition.shares,
                            'Buy Price':closedPosition.buyPrice, 
                            'Sold Price':closedPosition.soldPrice}, 
                           index=[closedPosition.soldDate])
        results = results.append(tmp)
        
    results.index.names = ['Date']
    results = results.reset_index()
    results = results.set_index(['Date','Symbol','Trade Type'])
    
    return results
        
    ### Returns a dataframe of trade results
def getUsedBuySignals():
## Lets create a dataframe to evaluate results using pandas
    
    
    usedBuySignals = pd.DataFrame(columns=['symbol'])
    
    for closedPosition in closedPositions:
        
        tmp_signals = pd.DataFrame({'symbol':closedPosition.symbol}, index=[closedPosition.buyDate])
        usedBuySignals = usedBuySignals.append(tmp_signals)

    usedBuySignals.index.names = ['Date']

    return usedBuySignals
    
## LOOP TO BUY STOCK USING BUY SIGNALS COMPUTED


from datetime import date as dateClass
import numpy as np

buyList = []

positions = {}
soldPositions = []
closedPositions = []
plan = Plan()
broker = Broker(plan)
report = Report()
balance = Balance()


# Go through all stock dates
irow = 0
maxSellDate = stocks['MSFT'].idxmax(axis=0).high
for date, row in stocks['MSFT'].iterrows():
        sellDate = stocks['MSFT'].index[irow+period-1]
        if(sellDate>maxSellDate):
            break
        
        balance.setCheckPoint(date, stocks)
        # Sell any stocks where date has expired
        
        # Check for any buy signals for this date and add to our buy list
        if(date.strftime('%Y-%m-%d') in buySignals.index ):
           
           # for idx, row in buySignals.loc[date].iterrows():
            for row in buySignals.loc[[date]].iterrows():
                symbol=row[1][0]
                if (symbol not in buyList) and (symbol not in positions):
                    buyList.append(symbol)
        
        # Buy any stocks in our buy list
        for symbol in buyList:
            data = stocks[symbol].loc[date]
            data.sellDate = sellDate
            broker.checkBuys(symbol, date, data)
        for symbol, position in positions.items():
            data = stocks[symbol].loc[date]
            broker.checkSells(position, symbol, date, data)
        for symbol in soldPositions:
            del positions[symbol]
            
       
        
        buyList=[]
        soldPositions = []
        
        irow +=1          
results = getResults()