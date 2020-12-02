import sqlite3 as sq3
import pdb
import numpy as np
import csv
import math

class DataLoader: 
    def __init__(self, data_path='../../noaa-data.db'): #, csv_filename='data/raw_data.csv'
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
        # self.csv_filename = csv_filename
        self.pos_key_removed = False


    def get_robot_keys_at_location(self, location='Gulf of Mexico'): # all robot_keys (trajectories) from gulf of mexico by default
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        cursor.execute("SELECT DISTINCT robot_key from robots where location=?",(location,))
        datasets = cursor.fetchall()
        datasets = [x[0] for x in datasets]
        return datasets
        '''
        [1, 2, 13, 29, 35, 36, 38, 39, 51, 52, 54, 55, 58, 64, 65, 66, 67, 70, 72, 78, 90, 159, 193, 194, 202, 204, 205, 
        206, 207, 208, 209, 210, 211, 213, 214, 215, 216, 217, 218, 237, 245, 247, 248, 249, 250, 251, 252, 253, 254, 255, 
        256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 292, 530, 531, 532, 547, 548, 549, 550, 551, 554, 555, 
        556, 557, 558, 560, 562, 564, 614, 615, 620, 621, 622, 623, 624, 625, 627, 628, 629, 630, 631, 632, 633, 634, 635, 
        753, 755, 757, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 777, 778, 779, 780, 781, 782, 783, 
        786, 849, 905, 906, 911, 934]
        '''


    # all starting position_keys of each dataset of a given location
    # num of datasets is at most 125
    def get_all_starting_positions(self, num_of_datasets=8, location='Gulf of Mexico'):
        datasets = self.get_robot_keys_at_location(location)        
        robot_map_starting = []
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        for i in range(num_of_datasets):
            robot = datasets[i]
            cursor.execute('select x.position_key from (position inner join robots on robots.robot_key==position.robot_key) as x where x.robot_key=? limit 1', (robot,))
            robot_map_starting.append(cursor.fetchone()[0])            
            # this takes really long if num_of_datasets is large
        robot_map_starting = dict(zip(datasets,robot_map_starting))
        return robot_map_starting


    # load a part of time series data
    def load_raw_data(self):
        connector = sq3.connect(self.data_path)
        cursor = connector.cursor()
        datasets = self.get_all_starting_positions()
        del datasets[1]
        del datasets[2]
        del datasets[13] #remove these later
        print(datasets)
        # BE MINDFUL OF DATASET (DO NOT MIX DIFFERENT DATASETS AS ONE TIME-SERIES SEQUENCE)
        trajectories = [k for k,v in datasets.items()]
        starting_positions = [v for k,v in datasets.items()]
        del starting_positions[-1]
        ending_positions = [v-1 for k,v in datasets.items()]
        del ending_positions[0]
        # can only get num_of_datasets-1 trajectories since last trajectory is only for determing ending position

        for traj in range(len(starting_positions)):
            start_pos = starting_positions[traj]
            end_pos = ending_positions[traj]
            cursor.execute("""SELECT position_key,time_of_day,latitude,longitude,depth 
                            FROM position WHERE position_key>=? AND position_key<=? 
                            AND position_key%100=0 ORDER BY time_of_day ASC""",(start_pos,end_pos))
            result = cursor.fetchall()

            # trim the data, remove when time difference with the next is less than 10 minutes
            min_diff = 600 # 2*ten minutes in unix
            list_del = []
            for i in range(len(result)-1):
                time_diff = result[i+1][1]-result[i][1]
                if(time_diff <= min_diff):
                    list_del.append(result[i])

            for el in list_del:
                result.remove(el)

            # fetch sensor data: conductivity, density, temperature, salinity
            # did not select pressure (why??)
            print("positions:",len(result))
            if(len(result)>=1500):
                continue
            for i in range(len(result)):
                fetch_data_cmd = "SELECT x.type_of_data, x.value FROM (sensor_data inner join position on position.position_key==sensor_data.position_key) as x WHERE x.position_key=? LIMIT 4"
                position_key = result[i][0]
                cursor.execute(fetch_data_cmd, (position_key,))
                sensor_values = []
                obj = cursor.fetchall()
                print(obj)
                if(len(obj) != 4):
                    continue
                elif(obj[0][0]!='conductivity' or obj[1][0]!='density' or obj[2][0]!='temperature' or obj[3][0]!='salinity'):
                    continue
                else:
                    sensor_values = [x[1] for x in obj]

                temp = list(result[i])
                temp.extend(sensor_values)
                result[i] = temp

            self.raw_data = result
            self.output_csv('mexico/traj_'+str(trajectories[traj])+'.csv')

        cursor.close()
        connector.close()
        # self.raw_data = result
        print("Done loading 5 trajectories")


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
        

    def output_csv(self, csv_filename):     #not necessary, might make it slower. just load the data from the array
        if not self.raw_data:
            self.load_raw_data()
        with open('data/'+csv_filename, 'w') as csvfile:
            csv_out = csv.writer(csvfile)
            if self.pos_key_removed:
                csv_out.writerow(['unix_time','latitude','longitude','depth','conductivity','density','temperature','salinity'])
            else:
                csv_out.writerow(['position_key','unix_time','latitude','longitude','depth','conductivity','density','temperature','salinity'])
            csv_out.writerows(self.raw_data)

        print("done with csv")
