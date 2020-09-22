import numpy as np

class TsganWrapper:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.dataX = []

    def MinMaxScaler(self): # normalize the data
        numerator = self.raw_data - np.min(self.raw_data, 0)
        denominator = np.max(self.raw_data, 0) - np.min(self.raw_data, 0)
        self.raw_data = numerator / (denominator + 1e-7)

    def build_dataset(self, seq_length=40):
        self.MinMaxScaler()
        self.dataX = []
        for i in range(len(self.raw_data) - seq_length):
            _x = self.raw_data[i:i+seq_length]
            self.dataX.append(_x)
        # Mix Data (to make it similar to i.i.d)
        np.random.shuffle(self.dataX)


