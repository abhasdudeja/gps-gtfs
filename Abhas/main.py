'''
Class to construct GPS data into Blocks.

'''
'''
Author: Abhas Dudeja
Date: 8th December 2023
'''

import pandas as pd
from math import radians, cos, sin, asin, sqrt
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Polygon
import random

class gps_data_utils:
    '''
    set previous lat long in the df
    '''
    @staticmethod
    def add_prev_latlong(df):
        df['prev_lat'] = df['lat'].shift(1).fillna(df['lat'])
        df['prev_lon'] = df['lon'].shift(1).fillna(df['lon'])
        return df

    '''
    Function to add date, time, and month from the timestamp column in the DF
    Return: DF
    '''
    @staticmethod
    def add_date_time_month_df(df):
        df = gps_data_utils.validate_mandatory_cols(df)
        col = 'timestamp'
        df['date'] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        df['month'] = pd.to_datetime(df[col]).dt.month
        df['time'] = pd.to_datetime(df[col]).dt.time
        df = df.sort_values(by=['vehicleid','timestamp']).reset_index(drop=True)
        return df
    '''
    Function to calculate haversine distance (m or km) between two lat-long coordinates.
    Return: distance rounded to 2 decimal digits
    '''
    @staticmethod
    def dist_calc_haversine_km(row):
        lon1, lat1, lon2, lat2 = map(radians, [row['lon'], row['lat'], row['prev_lon'], row['prev_lat']])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        km = 6367 * c
        return round(km,4)

    '''
    Function to calculate Geodesic distance (m or km) between two lat-long coordinates.
    Return: distance rounded to 2 decimal digits
    '''

    @staticmethod
    def dist_calc_geodesic_km(row):
        lat1,lon1 = row['lat'],row['lon']
        lat2,lon2 = row['prev_lat'],row['prev_lon']
        coord1 = (lat1,lon1)
        coord2 = (lat2,lon2)
        dist = geodesic(coord1,coord2).kilometers
        return round(dist,4)

    '''
    Read and return the CSV data.
    Return: Pandas DataFrame
    '''
    @staticmethod
    def read_data(file,cols,date_col,d_format):
        return pd.read_csv(file,usecols=cols,parse_dates=date_col,date_format=d_format)

    '''
    Validate columns exists in the GPS file.
    Mandatory Columns (timestamp, lat, lon, speed, vehicleid)
    '''
    @staticmethod
    def validate_mandatory_cols(df,ts = 'timestamp',la = 'lat',lo = 'lon',sp = 'speed',vid = 'vehicleid'):
        mandatory_cols = ['timestamp','lat','lon','speed','vehicleid']
        list_of_cols = df.columns
        if mandatory_cols[0] not in list_of_cols:
            # Remap Timestamp here
            df = gps_data_utils.remap_col(df,ts,mandatory_cols[0])
        if mandatory_cols[1] not in list_of_cols:
            # Remap Latitude here
            df = gps_data_utils.remap_col(df,la,mandatory_cols[1])
        if mandatory_cols[2] not in list_of_cols:
            # Remap Longitude here
            df = gps_data_utils.remap_col(df,lo,mandatory_cols[2])
        if mandatory_cols[3] not in list_of_cols:
            # Remap Speed here
            df = gps_data_utils.remap_col(df,sp,mandatory_cols[3])
        if mandatory_cols[4] not in list_of_cols:
            # Remap Speed here
            df = gps_data_utils.remap_col(df,vid,mandatory_cols[4])
        return df
    
    '''
    Function to rename columns from old name to new name in Python
    '''
    @staticmethod
    def remap_col(df,old_name,new_name):
        col_set = {old_name:new_name}
        df.rename(columns=col_set,inplace=True)
        return df
    
    '''
    Validate column headers for the GPS file exists.
    Optional Columns (elevation, enginestatus, gps_reason)
    '''
    @staticmethod
    def validate_optional_cols(df):
        optional_cols = ['elevation','enginestatus','gps_reason']
        list_of_cols = df.columns
        if optional_cols[0] not in list_of_cols:
            print("no - elevation")
        if optional_cols[1] not in list_of_cols:
            print(" no - enginer")
        if optional_cols[2] not in list_of_cols:
            print("no - reason")
        return df
    
    @staticmethod
    def df_to_gdf(df,lat='lat',lon='lon',crs = 'EPSG:4326'):
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon], df[lat], crs=crs), crs=crs
        )
    '''
    Using a set of input points, create a depot boundary.
    Optional: you can also give buffer parameter in arc degrees for buffer, default 0.
    '''
    @staticmethod
    def set_depot_boundary(points_seq,buffer = 0):
        polygon = Polygon([[p.y,p.x] for p in points_seq])
        polygon_buffer = polygon.buffer(buffer,single_sided=True)
        return gpd.GeoDataFrame(geometry=[polygon_buffer],crs='EPSG:4326')

    @staticmethod
    def check_veh_within_depot(df,depot):
        gdf = gps_data_utils.df_to_gdf(df)
        results = gpd.sjoin(gdf,depot,how='left',predicate='within')
        gdf['indepot'] = results['index_right'].notna()
        return pd.DataFrame(gdf.drop(columns='geometry'))
    
    @staticmethod
    def rgb_to_hex():
        color = random.choices(range(150),k=3)

        r = color[0]
        g = color[1]
        b = color[2]
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
