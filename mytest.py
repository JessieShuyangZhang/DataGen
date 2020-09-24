# from data_loader import DataLoader

# dl = DataLoader('converted_loc.csv')
# # dl.load_raw_data()
# dl.output_csv()
# print(dl.raw_data[0:5])

import numpy as np 
from tsgan_wrapper import TsganWrapper
x = np.loadtxt('converted_loc.csv', delimiter=',', skiprows=1)
# print(x[0:10])
wrapper = TsganWrapper(x)
wrapper.build_dataset()
print(wrapper.dataX[0:2])






'''
import datetime
timestamp = datetime.datetime.fromtimestamp(1500000000)
print(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
'''