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

# A thought that occur!
* We can use the indicators with various other information to determine when was the best time to buy and sell the stocks.
  Then based on the learning, we can train the RL agent to perform the learning
"""


# !pip install nsepython
# !pip install plotly
import datetime
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
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
        self.start_data = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%d-%m-%Y")


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

    def generate_macd_indicators(self):
        """
        generate the macd indicator for the stocks
        :return:
        """
        macd_results = indicators.get_macd(self.quotes_list, 12, 26, 9)
        self.macd_data = [r.macd for r in macd_results]


    def generate_bolinger_band_indicators(self):
        """
        generate the bolinger band indicator for the stocks
        :return:
        """
        self.bolinger_band_data = {}
        bolinger_band_results = indicators.get_bollinger_bands(self.quotes_list, 20, 2)
        self.bolinger_band_data["sma"] = [r.sma for r in bolinger_band_results]
        self.bolinger_band_data["lower_band"] = [r.lower_band for r in bolinger_band_results]
        self.bolinger_band_data["upper_band"] = [r.upper_band for r in bolinger_band_results]


    def generate_rsi_indicators(self):
        """
        generate the rsi indicator for the stocks
        :return:
        """
        rsi_results = indicators.get_rsi(self.quotes_list, 14)
        self.rsi_data = [r.rsi for r in rsi_results]


    def generate_indicators(self):
        """
        Generate various indicators from the captured stock data
        :return:
        """
        self.update_quotes()
        self.generate_macd_indicators()
        self.generate_bolinger_band_indicators()
        self.generate_rsi_indicators()

        self.all_inds = pd.DataFrame(self.bolinger_band_data, dtype=np.float32)
        self.all_inds['rsi'] = self.rsi_data
        self.all_inds['macd'] = self.macd_data
        self.all_inds.dropna(inplace=True)
        self.all_inds.reset_index(drop=True, inplace=True)
        self.all_inds = self.all_inds[::-1]
        return


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






# if __name__ == '__main__':
#     stock_obj = StockData("SBIN")
#     stock_obj.set_history_duration(365)
#     stock_obj.fetch_data()
#     stock_obj.plot_data() # this is needed only to visualize the data
#     stock_obj.generate_indicators()



class StockTrajectory:
    def __init__(self, stock_name):
        """
        Initialize the stock data object
        :param stock_name: stock whose data needs to monitored
        """
        self.stock_data_obj = StockData(stock_name)
        self.num_trading_days = 0


    def set_time_frame(self, duration_in_days=365):
        """
        set the past duration to capture the data
        :param duration_in_days: durations for which history data needs to be captured
        :return:
        """
        self.duration_in_days = duration_in_days


    def set_total_capital(self, capital_amount=100000):
        """
        set the total capital amount available. Either it is completely invested or completely withdrawn
        :param capital_amount: total capital
        :return:
        """
        self.capital_amount = capital_amount


    def process_data(self):
        """
        Process the data from the stock
        :return:
        """
        self.stock_data_obj.set_history_duration(self.duration_in_days)
        self.stock_data_obj.fetch_data()
        self.stock_data_obj.generate_indicators()
        self.num_trading_days = self.stock_data_obj.all_inds.shape[0]
        print(self.num_trading_days)


    def reset(self):
        """
        set the reference to the oldest record available
        :return:
        """
        self.record_idx = self.num_trading_days - 1
        self.capital_invested = 0
        self.nav_at_purchase = 0
        self.quantity = 0

        return self.stock_data_obj.all_inds.iloc[self.record_idx]


    def update_stock_params(self):
        """
        For the current record index, set the parameters associated with the stock data
        :return:
        """
        self.high_price = self.stock_data_obj.stock_df['CH_TRADE_HIGH_PRICE'][self.record_idx]
        self.low_price = self.stock_data_obj.stock_df['CH_TRADE_LOW_PRICE'][self.record_idx]
        self.open_price = self.stock_data_obj.stock_df['CH_OPENING_PRICE'][self.record_idx]
        self.close_price = self.stock_data_obj.stock_df['CH_CLOSING_PRICE'][self.record_idx]
        self.close_price_next = self.stock_data_obj.stock_df['CH_CLOSING_PRICE'][self.record_idx-1]
        self.close_price_prev = self.stock_data_obj.stock_df['CH_CLOSING_PRICE'][min(self.num_trading_days-1, self.record_idx+1)]


    def invest_capital(self, capital:float):
        """
        Invest specified capital
        :param capital: capital to invest
        :return:
        """
        # total amount to invest is limited by self.capital_amount
        investable_capital = min(self.capital_amount, capital)
        # calculate the NAVs for the purchase
        self.nav_at_purchase = self.close_price
        # calculate the holding quantity
        self.quantity = int(investable_capital / self.nav_at_purchase)
        # calculate the capital invested
        self.capital_invested = (self.quantity * self.nav_at_purchase)
        # reduce the investable capital amount
        self.capital_amount -= self.capital_invested


    def withdraw_capital(self):
        """
        Currently we assume that we withdraw entire capital
        :return:
        """
        # reset the capital invested
        self.capital_invested = 0
        # increment the holding quantity
        # self.quantity = 0 <= this should not be done here as we are using this parameter to normalize the net return
        # set current NAV
        self.nav_at_sell = self.close_price


    def calc_reward(self):
        """
        calculate the reward due to our current action
        :return:
        """
        # next day closing price - current day closing price
        reward = self.close_price_next - self.close_price
        return reward


    def step(self, action:int):
        """
        take a step abd update the next state
        :param action: (SELL, HOLD, BUY) which maps to (-1, 0, 1)
        :return: reward calculated as next_day_closing - current_day_closing
        """
        # as we are considering single time lumpsum investment/withdrawal, buy/hold have same return.
        self.update_stock_params()

        # this can be modified late based on the use-case
        if action == 1: # BUY
            self.invest_capital(self.capital_amount)
        elif action == -1: # SELL
            self.withdraw_capital()

        reward = self.calc_reward()
        # scale reward to number to quantities purchased
        reward_scaled = reward * self.quantity

        #if selling action was performed
        if action == -1:
            #reset the stock quantity to zero as we would have sold our entire holding
            self.quantity = 0
            # reward calculation was based on (price_tomorrow - price_today).
            # In case of selling, this becomes (price_today - price_tomorrow)
            reward_scaled *= -1

        # decrement the record index to next date
        self.record_idx -= 1

        done = (self.record_idx==0)
        # print(done, self.record_idx)

        next_state = (self.stock_data_obj.all_inds.iloc[self.record_idx] if not done else None)

        return reward_scaled, next_state, done


if __name__ == '__main__':
    trajectory_obj = StockTrajectory("SBIN")
    trajectory_obj.set_time_frame(730)
    trajectory_obj.set_total_capital(100000)
    trajectory_obj.process_data()
    trajectory_obj.reset()

    cumulative_reward = 0
    return_trajectory = [100000]

    reward, next_state, done = trajectory_obj.step(1)
    print(reward)
    reward, next_state, done = trajectory_obj.step(1)
    print(reward)
    reward, next_state, done = trajectory_obj.step(-1)
    print(reward)
    reward, next_state, done = trajectory_obj.step(-1)
    print(reward)
    return_trajectory.append(return_trajectory[-1]+reward)
    cumulative_reward += reward
    done = False
    while not done:
        reward, next_state, done = trajectory_obj.step(0)
        return_trajectory.append(return_trajectory[-1] + reward)
        cumulative_reward += reward

    plt.plot(return_trajectory)
    plt.show()





