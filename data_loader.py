import sqlite3 as sq3
# import pandas as pd 
import numpy as np
import csv
import math

class DataLoader: 
    def __init__(self, csv_filename='data/raw_data.csv', data_path='../../noaa-data.db'):
        '''
        load raw data from noaa database. converts location from lon-lat into x-y coords
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
        self.pos_key_removed = False

    # load a part of time series data
    def load_raw_data(self):
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        # all locations are from "Gulf of Mexico"
        # cursor.execute("SELECT position_key,time_of_day,latitude,longitude,depth FROM position WHERE position_key%100=0 AND position_key<=100000 ORDER BY time_of_day ASC")
        cursor.execute("""SELECT p.position_key,p.time_of_day,p.latitude,p.longitude,p.depth 
                        FROM position p, robots r 
                        WHERE p.robot_key=r.robot_key 
                        AND p.position_key%100=0 
                        AND p.position_key<=100000 
                        AND r.location='Gulf of Mexico' 
                        ORDER BY p.time_of_day ASC""")

        # cursor.execute("""SELECT p.position_key,p.time_of_day,p.latitude,p.longitude,p.depth 
        #                 FROM position p, robots r 
        #                 WHERE p.robot_key=r.robot_key 
        #                 AND p.position_key%100=0 
        #                 AND p.position_key>100000 
        #                 AND p.position_key<=200000 
        #                 AND r.location='Gulf of Mexico' 
        #                 ORDER BY p.time_of_day ASC""")
        result = cursor.fetchall()
        
        # trim the data, remove when time difference with the next is less than 10 minutes
        min_diff = 600 # ten minutes in unix
        list_del = []

        for i in range(len(result)-1):
            time_diff = result[i+1][1]-result[i][1]
            if(time_diff <= min_diff):
                list_del.append(result[i])

        for el in list_del:
            result.remove(el)   

        # fetch sensor data: conductivity, density, temperature, salinity
        for i in range(len(result)):
            fetch_data_cmd = "SELECT value FROM sensor_data WHERE position_key=? LIMIT 4"
            position_key = result[i][0]
            cursor.execute(fetch_data_cmd, (position_key,))
            sensor_values = []
            for x in range(4):
                sensor_values.append(cursor.fetchone()[0])
            temp = list(result[i])
            temp.extend(sensor_values)
            result[i] = temp

        cursor.close()
        connector.close()
        self.raw_data = result
        print("Done loading data")


    def convert_location(self):
        if not self.raw_data:
            print("loading data before converting location...")
            self.load_raw_data()
        # convert every pair of lat-lon to x-y coords
        # x = r*(lambda-lambda_0)*cos(phi_1), take phi_1 to be 29
        # y = r*(phi-phi_0)
        lambda_0 = self.raw_data[0][3]*math.pi/180  # longitude in rad
        phi_0 = self.raw_data[0][2]*math.pi/180     # latitude in rad
        phi_1 = 29*math.pi/180   # hard-coded, not sure; the range in which the conversion is valid
        cos_phi1 = math.cos(phi_1*math.pi/180)
        for i in range(len(self.raw_data)):
            lambda_ = self.raw_data[i][3]*math.pi/180
            phi_ = self.raw_data[i][2]*math.pi/180
            earth_r = 3958.8       # in miles
            x = earth_r * (lambda_ - lambda_0) * cos_phi1
            y = earth_r * (phi_ - phi_0)
            # replace (lat,lon) with (x,y)
            self.raw_data[i][2] = x
            self.raw_data[i][3] = y

        print("Done converting location") 


    def convert_time(self): # let first row's time be base time t0, subtract it from all subsequent rows
        if not self.raw_data:
            print("loading data before converting time...")
            self.load_raw_data()
        t0 = self.raw_data[0][1]
        for i in range(len(self.raw_data)):
            self.raw_data[i][1] -= t0
        
        print("Done converting time")
        
    
    def del_position_key(self): #remove the first column 'position_key' from raw_data
        if not self.raw_data:
            self.load_raw_data()
        for i in self.raw_data:
            del i[0]
        self.pos_key_removed = True
        

    def output_csv(self):     #not necessary, might make it slower. just load the data from the array
        if not self.raw_data:
            self.load_raw_data()
        with open('data/'+self.csv_filename, 'w') as csvfile:
            csv_out = csv.writer(csvfile)
            if self.pos_key_removed:
                csv_out.writerow(['unix_time','latitude','longitude','depth','conductivity','density','temperature','salinity'])
            else:
                csv_out.writerow(['postion_key','unix_time','latitude','longitude','depth','conductivity','density','temperature','salinity'])
            csv_out.writerows(self.raw_data)

        print("done with csv")
