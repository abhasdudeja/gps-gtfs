'''
Class to construct GPS data into Blocks.

'''
'''
Author: Abhas Dudeja and Durga Lekshmi
Date: 19th December 2023

'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
from geopy.distance import geodesic
from osmnx import distance, utils_graph, settings, graph
settings.use_cache = True
settings.cache_only_mode = True
import matplotlib.pyplot as plt

class gps_preprocessor: # GPS Data Processor

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
            df = gps_preprocessor.remap_col(df,ts,mandatory_cols[0])
        if mandatory_cols[1] not in list_of_cols:
            # Remap Latitude here
            df = gps_preprocessor.remap_col(df,la,mandatory_cols[1])
        if mandatory_cols[2] not in list_of_cols:
            # Remap Longitude here
            df = gps_preprocessor.remap_col(df,lo,mandatory_cols[2])
        if mandatory_cols[3] not in list_of_cols:
            # Remap Speed here
            df = gps_preprocessor.remap_col(df,sp,mandatory_cols[3])
        if mandatory_cols[4] not in list_of_cols:
            # Remap Speed here
            df = gps_preprocessor.remap_col(df,vid,mandatory_cols[4])
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
    
    '''
    Function to add date, time, and month from the timestamp column in the DF
    Return: DF
    '''
    @staticmethod
    def add_date_time_month_df(df):
        df = gps_preprocessor.validate_mandatory_cols(df)
        col = 'timestamp'
        df['date'] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        df['month'] = pd.to_datetime(df[col]).dt.month
        df['time'] = pd.to_datetime(df[col]).dt.time
        df = df.sort_values(by=['vehicleid','timestamp']).reset_index(drop=True)
        return df
    
    '''
    Using a set of input points, create a depot boundary.
    Optional: you can also give buffer parameter in arc degrees for buffer, default 0.
    '''
    @staticmethod
    def set_depot_boundary(points_seq,buffer = 0):
        polygon = Polygon([[p.y,p.x] for p in points_seq])
        polygon_buffer = polygon.buffer(buffer,single_sided=True)
        centroid_point = Point(polygon.centroid.x, polygon.centroid.y)
        return gpd.GeoDataFrame(geometry=[polygon_buffer],crs='EPSG:4326'), centroid_point 
    
    '''
    Set previous lat long in the df
    '''
    @staticmethod
    def add_prev_latlong(df):
        df['prev_lat'] = df['lat'].shift(1).fillna(df['lat'])
        df['prev_lon'] = df['lon'].shift(1).fillna(df['lon'])
        return df
    
    '''
    Read the DataFrame and return GeoDataFrame 
    '''
    @staticmethod
    def df_to_gdf(df,lat='lat',lon='lon',crs = 'EPSG:4326'):
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat], crs=crs), crs=crs)
    
    '''
    Check within the depot or not
    '''
    @staticmethod
    def check_veh_within_depot(df,depot):
        gdf = gps_preprocessor.df_to_gdf(df)
        results = gpd.sjoin(gdf,depot,how='left',predicate='within')
        gdf['indepot'] = results['index_right'].notna()
        return pd.DataFrame(gdf.drop(columns='geometry'))
    
    '''
    Function to calculate Geodesic distance (km) between two lat-long coordinates.
    Return: distance rounded to 4 decimal digits
    '''
    @staticmethod
    def cal_dist_prev_point(df):
        df['dist_prev_point'] = df.apply(lambda row: round(geodesic((row['lat'], row['lon']), (row['prev_lat'], row['prev_lon'])).kilometers, 4), axis=1)
        return df
    
    '''
    Speed flag filter
    '''
    @staticmethod
    def calculate_speed_flag(df, error_factor):
        df['dist_prev_point'] = df.apply(lambda row: round(geodesic((row['lat'], row['lon']), (row['prev_lat'], row['prev_lon'])).kilometers, 4), axis=1)
        df['timediff'] = df.groupby(by=['vehicleid', 'date'])['timestamp'].transform(lambda x: x.diff().dt.total_seconds() / 3600)
        df['velocity_gps'] = df['dist_prev_point'] / df['timediff'].fillna(0)
        df['velocity_vehicle'] = (df['speed'] + df['speed'].shift(1)) / 2
        df['speed_flag'] = (df['velocity_vehicle'] != 0) & (abs((df['velocity_gps'] - df['velocity_vehicle']) / df['velocity_vehicle']) < error_factor)
        return df
    
    @staticmethod
    def filterby_speed(df):
        return df[df['speed_flag']]
    
    @staticmethod
    def blank_filter(df):
        '''
        Row-by-row check of gps data
        '''
        return df
    
    @staticmethod
    def get_network(df):
        """
            Get the network graph based on the given DataFrame.
            Args:
                df: The DataFrame containing latitude and longitude coordinates.
            Returns:
                The network graph generated from the bounding box of the DataFrame.
            Examples:
                >>> df = pd.DataFrame(...)
                >>> get_network(df)
                ... # Output: The network graph
        """
        min_lat = df['lat'].mix()
        max_lat = df['lat'].max()
        min_lon = df['lon'].mix()
        max_lon = df['lon'].max()
        bounding_box = (min_lat, min_lon, max_lat, max_lon)
        return graph.graph_from_bbox(bounding_box[0], bounding_box[2], bounding_box[1], bounding_box[3], network_type='drive', simplify=True)
    
    @staticmethod
    def interpolate_dutycycle(df):
        '''
        interpolate 
        '''
        return df
    
    '''
    Returns the distance between any two points
    '''
    @staticmethod
    def ret_geo_dist_km(p1y,p1x,p2y,p2x):
        return geodesic((p1y,p1x),(p2y,p2x)).kilometers

    '''
    Returns a df of size n x 3 (count, latitude, longitude)
    '''
    @staticmethod
    def depot_locator(df):
        df_gpd = gps_preprocessor.df_to_gdf(df)
        grouped = df_gpd.groupby('geometry').size().sort_values(ascending=False).reset_index(name='Count')
        factor = 0.01 if len(grouped) < 1500000 else 0.005
        end= int(len(grouped)*factor)
        counts = []
        other_counts = 0
        for i in range(len(grouped)):
            if len(counts) < end:
                if i > 0:
                    distance = gps_preprocessor.ret_geo_dist_km(grouped['geometry'][i].y,grouped['geometry'][i].x,grouped['geometry'][i-1].y,grouped['geometry'][i-1].x)
                    if distance > 0.5:
                        counts.append([[grouped['geometry'][i].y,grouped['geometry'][i].x],grouped['Count'][i]])
                    else:
                        other_counts += grouped['Count'][i]
                else:
                    counts.append([[grouped['geometry'][i].y,grouped['geometry'][i].x],grouped['Count'][i]])
            else:
                other_counts += grouped['Count'][i]
        counts = pd.DataFrame(counts,columns=['coords','count']).sort_values(by='count',ascending=False).reset_index()
        counts[['lat', 'lon']] = pd.DataFrame(counts['coords'].tolist(), columns=['lat', 'lon'])
        counts['lat'] = counts['lat'].astype(float)
        counts['lon'] = counts['lon'].astype(float)
        counts_gpd = gpd.GeoDataFrame(counts,geometry=gpd.points_from_xy(counts['lon'],counts['lat']),crs='EPSG:4326')
        for i in range(len(counts_gpd)):
            if i in counts_gpd.index:
                point1 = counts_gpd.iloc[i]
                for j in range(len(counts_gpd)):
                    if j in counts_gpd.index:
                        point2 = counts_gpd.iloc[j]
                        distance = gps_preprocessor.ret_geo_dist_km(point1['geometry'].y,point1['geometry'].x,point2['geometry'].y,point2['geometry'].x)
                        if 0.0 < distance < 0.5:
                            counts_gpd.at[counts_gpd[counts_gpd['index'] == point1['index']].index[0], 'count'] += counts_gpd.at[j, 'count']
                            counts_gpd = counts_gpd.drop(j)
                            counts_gpd = counts_gpd.reset_index(drop=True)
        counts_gpd = counts_gpd.sort_values(by='count', ascending=False).reset_index(drop=True)
        return pd.DataFrame(counts_gpd.drop(columns='geometry'))

    @staticmethod
    def exportfor_routeenergy(df):
        '''
        function helps export files for route energy modelling, i.e., Block Schedule and Duty Cycles.
        '''
        return df
    
    '''
    Function to add dead-head row of depot location if ( start or end ) are not in depot location 
    '''
    @staticmethod
    def add_dh_trips(df, centroid_point):
        depot_lat = centroid_point.y
        depot_lon = centroid_point.x
        grouped_df = df.groupby(by=['vehicleid'])
        all_trips = []
        for main_key, group in grouped_df:
            flag = False
            curr_start = None
            end = None
            for index, row in group.iterrows():
                if row['indepot'] is not True and curr_start is None:
                    flag = True
                    curr_start = index
                elif flag and row['indepot'] is True and curr_start is not None and end is None:
                    end = index
                    trip = df.loc[curr_start:end]
                    new_row = trip.iloc[0].copy()
                    new_row['lat'], new_row['lon'] = depot_lat, depot_lon
                    trip = pd.concat([pd.DataFrame([new_row]), trip])
                    trip = pd.concat([trip, pd.DataFrame([new_row])])
                    all_trips.append(trip)
                    curr_start = None
                    end = None
        return pd.concat(all_trips, ignore_index=True)
    
    '''
    Convert subset df (trip) into LineString and simplify it based on tolerance.
    '''
    @staticmethod
    def simplfy_trip_byvehicle(df, tolerance):
        gdf = gps_preprocessor.df_to_gdf(df)
        gdf_grouped = gdf.groupby(by=['vehicleid'])
        line_gdf = []
        for vehicleid, group in gdf_grouped:
            line = LineString(zip(group['lon'], group['lat']))
            simplified_line = line.simplify(tolerance)
            line_gdf.append(gpd.GeoDataFrame({'vehicleid': [group.iloc[0]['vehicleid']], 'geometry': [simplified_line]}, crs='EPSG:4326'))
        return gpd.GeoDataFrame(pd.concat(line_gdf, ignore_index=True),geometry='geometry', crs='EPSG:4326')
    
    @staticmethod
    def export_speed_flag_df(file, cols, date_col, d_format, depot, error_factor):
        df = gps_preprocessor.read_data(file,cols,date_col,d_format)
        df = gps_preprocessor.validate_mandatory_cols(df,'RowReferenceTime','lat','lon','Speed','UnityLicensePlate')
        df = gps_preprocessor.add_date_time_month_df(df)
        df = df.groupby(by=['vehicleid','date']).apply(gps_preprocessor.add_prev_latlong).reset_index(drop=True)
        df = gps_preprocessor.check_veh_within_depot(df,depot)
        df = gps_preprocessor.calculate_speed_flag(df, error_factor)
        df.to_csv('speed_flag_df.csv', index=False)

    
class gps_preprocessor_analysis:  # GPS Data Outputs

    '''
    Daily Aggregate Operational Distance - km
    '''
    @staticmethod
    def daily_agg_operational_dist_km(df,median_color,max_color,distance_color):
        plt.figure(figsize=(12, 6)) 
        graph_df = df.groupby('start_date').agg({'new_distance': 'sum'}).reset_index()
        graph_df['new_distance'] = graph_df['new_distance'].round(4)
        graph_df = graph_df.sort_values(by='start_date')
        plt.plot(graph_df['start_date'], graph_df['new_distance'], color=distance_color, label='Distance')
        median_distance = round(graph_df['new_distance'].median(), 1)
        plt.axhline(y=median_distance, color=median_color, linestyle='-', linewidth=2, label=f'Median: {median_distance:.1f} km')
        max_distance = round(graph_df['new_distance'].max(), 1)
        plt.axhline(y=max_distance, color=max_color, linestyle='-', linewidth=2, label=f'Maximum: {max_distance:.1f} km')
        plt.xlabel('Date')
        plt.ylabel('Distance (km)')
        plt.title('Daily Aggregated Operational Distance')
        plt.legend()
        plt.show()
        
    '''
    Daily Aggregate Operational Distance Distribution
    '''
    @staticmethod
    def daily_agg_operational_dist_distribution(df):
        dist_df = df.groupby('start_date').agg({'new_distance': 'sum'}).reset_index()
        dist_df['new_distance'] = dist_df['new_distance'].round(4)
        dist_df = dist_df.sort_values(by='start_date')
        max_dist = int(dist_df['new_distance'].max())
        if max_dist % 50 != 0:
            rounded_max_dist = ((max_dist + 49) // 50) * 50
        else:
            rounded_max_dist = max_dist
        bins = list(range(0, rounded_max_dist + 50, 50))  # Adjust this as per your requirement
        dist_df['distance'] = pd.cut(dist_df['new_distance'], bins, right=False)
        graph_df = dist_df.groupby('distance', observed=False).count().reset_index()[['distance', 'new_distance']].rename(columns={'new_distance': 'count_of_date'})
        graph_df['distance'] = graph_df['distance'].astype(str)
        graph_df['distance'] = graph_df['distance'].str.replace(', ', '-', regex=False)
        graph_df['distance'] = graph_df['distance'].str.replace(r'\[|\)', '', regex=True)
        total_count = graph_df['count_of_date'].sum()
        graph_df['percentage'] = (graph_df['count_of_date'] / total_count) * 100
        graph_df['percentage'] = graph_df['percentage'].round(2)
        graph_df['combined_labels'] = graph_df['distance'].astype(str) + '\n' + graph_df['count_of_date'].astype(str)
        plt.figure(figsize=(6, 6))
        bars = plt.bar(graph_df['combined_labels'], graph_df['count_of_date'], color='blue', width=0.6)
        plt.xticks(graph_df['combined_labels'], ha='center', va='bottom', position=(0, -0.06), color='black')
        for bar, percentage in zip(bars, graph_df['percentage']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{percentage}%', ha='center', va='bottom')
        plt.xlabel('Daily Agg Ops Distance')
        plt.title('Daily Aggregate Operational Distance Distribution')
        plt.show()