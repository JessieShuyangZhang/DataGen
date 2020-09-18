import sqlite3 as sq3
# import pandas as pd 
import numpy as np
import csv
import datetime

class DataLoader: 
    def __init__(self, data_path='../../../../media/raid/jessiezhang/noaa-data.db'):
        self.data_path = data_path
        self.raw_data = []

    # load a part of time series data
    def load_raw_data(self):
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        cursor.execute("SELECT position_key,time_of_day,latitude,longitude,depth FROM position WHERE position_key%100=0 AND position_key<=1000000 ORDER BY time_of_day ASC")
        result = cursor.fetchall()
        

        # trim the data 
        # remove when time difference with the next is less than 10 minutes
        min_diff = 600 # ten minutes in unix
        list_del = []

        for i in range(len(result)-1):
            time_diff = result[i+1][1]-result[i][1]
            if(time_diff <= min_diff):
                list_del.append(result[i])

        for el in list_del:
            result.remove(el)   

        # fetch sensor data: conductivity, density, temperature, salinity
        # for el in result:
        for i in range(5):
            fetch_data_cmd = "SELECT value FROM sensor_data WHERE position_key=? AND type_of_data=? LIMIT 1"
            # position_key = el[0]
            position_key = result[i][0]
            cursor.execute(fetch_data_cmd, (position_key,'conductivity'))
            conductivity = cursor.fetchone()[0]
            cursor.execute(fetch_data_cmd, (position_key,'density'))
            density = cursor.fetchone()[0]
            cursor.execute(fetch_data_cmd, (position_key,'temperature'))
            temperature = cursor.fetchone()[0]
            cursor.execute(fetch_data_cmd, (position_key,'salinity'))
            salinity = cursor.fetchone()[0]
            temp = list(result[i])
            temp.extend([conductivity, density, temperature, salinity])
            result[i] = tuple(temp)
            

        cursor.close()
        connector.close()
        self.raw_data = result
        # return result

    def output_csv():
        pass

    


'''
import datetime
timestamp = datetime.datetime.fromtimestamp(1500000000)
print(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
'''