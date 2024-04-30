import numpy as np

from stock_data import StockData

class PeakExtraction:
    def __init__(self):
        pass

    def set_data(self, data:np.ndarray):
        """
        set the 1D data from which we need to extract peak
        :param data: 1D data
        :return:
        """
        self.data = data


    def calculate_peaks(self):
        """
        Calculate peak locations
        :return:
        """
        # get the length of data list
        L = len(self.data)
        # generate index list
        index_list = np.arange(L)
        # calculate the position of highs
        #self.high_indices = np.where([data[idx] >= data[idx - 1] and data[idx] >= data[idx + 1] for idx in index_list[1:-1]])[0] + 1
        # calculate the position of lows
        #low_indices = np.where([data[idx] <= data[idx - 1] and data[idx] <= data[idx + 1] for idx in index_list[1:-1]])[0] + 1
        #self.low_indices = np.array(sorted(low_indices.tolist() + [0, L-1]))




if __name__ == '__main__':
    stock_obj = StockData("SBIN")
    stock_obj.set_history_duration(365)
    stock_obj.fetch_data()
    high_data = stock_obj.stock_df['CH_TRADE_HIGH_PRICE']
    low_data = stock_obj.stock_df['CH_TRADE_LOW_PRICE']

    obj = PeakExtraction()
    obj.set_data(high_data.tolist())
    obj.calculate_peaks()
    pass

