"""
We target the large cap stocks for the analysis.
These large cap stocks will provide higher long term return but at a cost of higher volatality
In addition, we may consider one liquid fund which ensures low volatility but at cost of lower return.

ASSUMPTIONS:
Initially it can be assumed that the transaction (buy/sell) are instantaneous.
Also, it can be assumed that we can perform multiple buy/sell actions without any restrictions per day.
Later, we can try to introduce the latency of T+1 day for stock settling, which will implicitly
introduce a restriction of single buy/sell actions in a day.

OBJECTIVE 1:
We consider a single stock and the liquid fund.
Initially, the entire corpus can be assumed to be in liquid fund.
When we have 'BUY' trigger, we move the entire corpus from liquid fund to large cap stock.
When we have 'SELL' trigger, we move the corpus from large cap stock back to liquid fund.

OBJECTIVE 2:
Once we have a working solution for the prior objective, we can enable partial buy/sell.

OBJECTIVE 3:
Eventually, we can enable multiple large cap stock with partial allocations to each stock.
We may need to derive some parameter which can help us in prioritizing some stocks over other.

OBJECTIVE 4:
We can also add the time horizon, lumpsum/SIP scenarios into consideration.


APPROACH:
In order to solve any problem using Reinforcement Learning, we need STATE, ACTION and REWARD parameters.
In our case, ACTION space is discrete and well defined (BUY, HOLD, SELL).
REWARD is the % change in the price next day. RETURN can be computed as the overall change in the stock price.
However, the STATE space is quite unclear and probably need to be generated.
Based on some literatures and references, we may have to generate various measures whose collection can be considered
as the overall state. These measures can be:
 * Trading indicators: MACD, RSI, Bolinger Band, CCI
 * Minima over multiple time frames (2 months/4 months/6 months)
 Moving average over multiple time frames might be needed to capture the long term (trend) and the short term fluctuations.
 Bolinger band cross overs can be useful to confirm the BUY/SELL actions.

"""
!pip install nsepython
!pip install plotly
import datetime

from nsepython import *

import plotly.graph_objects as go

from stock_indicators import indicators, Quote # https://python.stockindicators.dev/indicators/#content

class StockData:
    """
    We use this class to get data for a given stock
    """
    def __init__(self, symbol="SBIN"):
        self.set_symbol(symbol)
        self.series = "EQ"


    def set_symbol(self, symbol):
        """
        Set the stock symbol. This symbol will be used to perform queries
        symbol: stock symbol
        """
        self.symbol = symbol


    def set_history_duration(self, days=365):
        """
        Based on the days parameter, we fetch historical data from nse. By default we perform query for 365 days
        """
        self.history_duration = days
        end_data = datetime.datetime.now().strftime("%d-%m-%Y")
        self.end_date = str(end_data)
        self.start_data = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%d-%m-%Y")


    def fetch_data(self):
        """
        Fetch the historic data associated with the stock symbol
        """
        self.stock_df = equity_history(self.symbol, self.series, self.start_data, self.end_date)
        #!!! We may need to normalize the mean to ZERO !!!


    def normalize_data(self):
        """
        Normalize the mean to ZERO
        :return:
        """
        self.stock_df['Close'] = self.stock_df['Close']


    def update_quotes(self):
        """
        Update the quote list needed for generating indicators
        :return:
        """
        datetime_list = []
        for date_str in self.stock_df['CH_TIMESTAMP']:
            year, month, date = re.findall("(\d+)-(\d+)-(\d+)", date_str)[0]
            datetime_list.append(datetime.datetime(int(year), int(month), int(date), 0, 0, 0))

        self.quotes_list = [ Quote(date, open, high, low, close, volume)
                             for date, open, high, low, close, volume
                             in zip(datetime_list, self.stock_df['CH_OPENING_PRICE'], self.stock_df['CH_TRADE_HIGH_PRICE'],
                                    self.stock_df['CH_TRADE_LOW_PRICE'], self.stock_df['CH_CLOSING_PRICE'], self.stock_df['CH_TOT_TRADED_QTY'])
                             ]

    def generate_indicators(self):
        """
        Generate various indicators from the captured stock data
        :return:
        """
        self.update_quotes()
        macd_results = indicators.get_macd(self.quotes_list, 12, 26, 9)
        self.macd_data = [r.macd for r in macd_results]


    def plot_data(self):
        """
        Plots the stock data
        """
        fig = go.Figure(data=[go.Candlestick(x=self.stock_df['CH_TIMESTAMP'],
                                             open=self.stock_df['CH_OPENING_PRICE'],
                                             high=self.stock_df['CH_TRADE_HIGH_PRICE'],
                                             low=self.stock_df['CH_TRADE_LOW_PRICE'],
                                             close=self.stock_df['CH_CLOSING_PRICE'])])

        fig.show()






if __name__ == '__main__':
    stock_obj = StockData("SBIN")
    stock_obj.set_history_duration(365)
    stock_obj.fetch_data()
    stock_obj.plot_data() # this is needed only to visualize the data
    stock_obj.generate_indicators()
