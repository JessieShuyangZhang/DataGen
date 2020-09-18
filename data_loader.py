import sqlite3 as sq3
# import pandas as pd 
import numpy as np
import csv

class DataLoader: 
    def __init__(self, csv_filename='raw_data.csv', data_path='../../../../media/raid/jessiezhang/noaa-data.db'):
        '''
        @type data_path: string
        @param data_path: the relative path to noaa database
        @type raw_data: list
        @param raw_data: an array of tuples that stores time series data
        @type csv_filename: string
        @param csv_filename: the name of the csv file storing the raw data
        '''
        
        self.data_path = data_path
        self.raw_data = []
        self.csv_filename = csv_filename

    # load a part of time series data
    def load_raw_data(self):
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        cursor.execute("SELECT position_key,time_of_day,latitude,longitude,depth FROM position WHERE position_key%100=0 AND position_key<=100000 ORDER BY time_of_day ASC")
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
        # prog = 0  # just to show progress
        for i in range(len(result)):
            fetch_data_cmd = "SELECT value FROM sensor_data WHERE position_key=? LIMIT 4"
            position_key = result[i][0]
            cursor.execute(fetch_data_cmd, (position_key,))
            sensor_values = []
            for x in range(4):
                sensor_values.append(cursor.fetchone()[0])
            temp = list(result[i])
            temp.extend(sensor_values)
            result[i] = tuple(temp)
            # print(prog)
            # prog+=1

        cursor.close()
        connector.close()
        self.raw_data = result

    def output_csv(self):
        if not self.raw_data:
            self.load_raw_data()
        with open(self.csv_filename, 'wb') as csvfile:
            csv_out = csv.writer(csvfile)
            csv_out.writerow(['postion_key','unix_time','latitude','longitude','depth','conductivity','density','temperature','salinity'])
            csv_out.writerows(self.raw_data)

        print("done with csv")
