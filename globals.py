import pandas_datareader.data as web
import datetime

stockSymbols = ['AMD']
startDate = datetime.datetime(2010, 1, 1)
endDate = datetime.datetime.now()

DATE_DATA = web.get_data_yahoo('MSFT',startDate,endDate)
