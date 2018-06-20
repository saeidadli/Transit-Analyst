"""
This module defines a Feed class and a PT_network class based on a GTFS feeds.
There is an instance attribute for every GTFS table (routes, stops, etc.),
which stores the table as a Pandas DataFrame,
or as ``None`` in case that table is missing.
The Feed class also has heaps of methods. For more information 
check Alex Raichev's GTFS toolkit.
I have added few more functionality to the Feed object so I can
use it to calculate the transit catchments.

"""

import tempfile
import time 
import shutil
import math
import sys

from pathlib import Path
from datetime import datetime, timedelta
from dateutil.parser import parse

import pandas as pd
import numpy as np
import geopandas as gp
import shapely.geometry as geom
import gtfstk as gt

from . import helpers as hp


class Feed(gt.Feed):
    def __init__(self, dist_units, agency=None, stops=None, routes=None,
        trips=None, stop_times=None, calendar=None, calendar_dates=None,
        fare_attributes=None, fare_rules=None, shapes=None,
        frequencies=None, transfers=None, feed_info=None, feed_segments=None,
        valid_date=None):
        
        self.dist_units = dist_units
        self.agency = agency
        self.stops = stops
        self.routes = routes
        self.trips = trips
        self.stop_times = stop_times
        self.calendar = calendar
        self.calendar_dates = calendar_dates
        self.fare_attributes = fare_attributes
        self.fare_rules = fare_rules
        self.shapes = shapes
        self.frequencies = frequencies
        self.transfers = transfers
        self.feed_info = feed_info
        self.feed_segments = feed_segments
        self.valid_date = valid_date
        
    @property
    def valid_day_of_week(self):
        self.valid_day_of_week = parse(valid_date).weekday('%A')

    
class pt_network:
    def __init__(
        self, 
        feed=None, 
        ptedge=None, 
        wedge=None,
        analysis_start=None, 
        analysis_end=None,
        transfer_duration=None,
        walk_speed_kph=None,
    ):
        
        self.feed = feed                                        #feed object is generated from read_gtfs
        self.ptedge = ptedge                                    #links stops by transit
        self.wedge = wedge                                      #links stops by walking 
        self.analysis_start = analysis_start                    #Analysis start time
        self.analysis_end = analysis_end                        #Analysis end time
        self.transfer_duration = transfer_duration
        self.walk_speed_kph = walk_speed_kph
        
    @property
    def analysis_start_sec(self):
        return gt.timestr_to_seconds(self.analysis_start)

    @property
    def analysis_end_sec(self):
        return gt.timestr_to_seconds(self.analysis_end)

    @property
    def analysis_duration_sec(self):
        return self.analysis_end_sec - self.analysis_start_sec

    @property
    def analysis_time_sec(self):                            #Middle point for our analysis
        return self.analysis_start_sec + self.analysis_duration_sec/2

    @property
    def analysis_time(self):
        return hp.sec2text(self.analysis_time_sec)

    @property
    def transfer_duration_sec(self):
        return self.transfer_duration * 60
        
"""
Functions about creating abundant access.
"""

def read_gtfs(
    path, 
    dt,   #date to validate feed upon, it can be like "Thrusday" or "20181201"
    dist_units=None):
    """
    Create a Feed instance from the given path and given distance units.
    The path should be a directory containing GTFS text files or a
    zip file that unzips as a collection of GTFS text files
    (and not as a directory containing GTFS text files).
    The distance units given must lie in :const:`constants.dist_units`
    Notes
    -----
    - Ignore non-GTFS files
    - Automatically strip whitespace from the column names in GTFS files
    - This is based on gtfstk library
    """
    gt_feed = gt.read_gtfs(path, dist_units)
    
    #Validate feed for an specific day (eigther a date or the day of week)========
    if not gt.valid_date(dt):
        dt = gt_feed.get_first_week()[parse(dt).weekday()]
       
    
    gt_feed = hp.validate_feed(gt_feed, dt)
    feed_dict = hp.feed_obj_to_dict(gt_feed) 
    
    feed_dict['valid_date'] = dt

    #calculate PT segments========================================================
   
    PT_links_df =  feed_dict['stop_times'].copy()
    #making sure trips are sorted by the trip sequence
    PT_links_df.sort_values(by=['trip_id', 'stop_sequence'], inplace = True)
    
    #converting the stop_times into pt links
    PT_links_df.rename(columns = {'arrival_time': 'o_time', 
                                  'stop_id': 'o_stop',
                                  'stop_sequence': 'o_sequence'}, inplace = True)  
    
    PT_links_df[['d_time', 'd_stop', 'd_sequence']] = PT_links_df[['o_time', 'o_stop', 'o_sequence']].shift(-1)
    
    PT_links_df = PT_links_df[PT_links_df['o_sequence'] < PT_links_df['d_sequence']].copy() #removes the last stops
    
    #Convert the time into seconds for easier time calculatins
    PT_links_df['o_time_sec'] = PT_links_df['o_time'].apply(hp.text2sec)
    PT_links_df['d_time_sec'] = PT_links_df['d_time'].apply(hp.text2sec)
    PT_links_df['duration'] = PT_links_df['d_time_sec'] - PT_links_df['o_time_sec']
    
    #Add route_id using the trips table
    PT_links_df = PT_links_df.merge(feed_dict['trips'])
        
    #Add route type in text format to the link dataset
    PT_links_df = PT_links_df.merge(feed_dict['routes'])

    route_type = {'0': 'Tram, Streetcar, Light rail',
                  '1': 'Subway, Metro',
                  '2': 'Rail',
                  '3': 'Bus',
                  '4': 'Ferry',
                  '5': 'Cable car',
                  '6': 'Gondola, Suspended cable car',
                  '7': 'Funicular'}

    PT_links_df['route_type'] = PT_links_df['route_type'].astype(str)
    PT_links_df['route_type'].replace(route_type, inplace = True)

    #add stop sequence to PT_links_df
    def stop_seq_for_trips(stop_times_df):
        """
        The objective is to create a dataframe of stop sequence for each trip
        The output format will be:
          first field is: trip_ids
          seocond field is: stop_ids separeated by comma in order of their sequence
        """

        def get_first_trip(group):
            stop_seq = ";".join(group['stop_id'].tolist())+";"
            trip_id = group['trip_id'].iat[0]
            trip_dict = {'stop_seq': stop_seq,
                         'trip_id': trip_id}
            return pd.DataFrame(trip_dict, index=[0])
        stop_seq_df = stop_times_df.groupby('trip_id').apply(get_first_trip).reset_index(drop=True)
        return stop_seq_df
    
    stop_seq_df = stop_seq_for_trips(feed_dict['stop_times'])
    PT_links_df = PT_links_df.merge(stop_seq_df)
    
    def remaining_stops(row):
        sid = row['o_stop']+";"
        seq = row['stop_seq']
        return seq.split(sid, 1)[-1]
    PT_links_df['stop_seq'] = PT_links_df.apply(remaining_stops, axis = 1)
    
    # add stops lat and lon
    PT_links_df = PT_links_df.merge(feed_dict['stops'][['stop_id', 'stop_lat', 'stop_lon']], 
                                    left_on='o_stop', 
                                    right_on='stop_id', 
                                    how='left').drop('stop_id', axis = 1) 
    PT_links_df.rename(columns = {'stop_lat': 'o_stop_lat',
                                  'stop_lon': 'o_stop_lon'},
                      inplace = True)
    PT_links_df = PT_links_df.merge(feed_dict['stops'][['stop_id', 'stop_lat', 'stop_lon']], 
                                    left_on='d_stop', 
                                    right_on='stop_id', 
                                    how='left').drop('stop_id', axis = 1)
    PT_links_df.rename(columns = {'stop_lat': 'd_stop_lat',
                                  'stop_lon': 'd_stop_lon'},
                      inplace = True)
    
    feed_dict['feed_segments'] = PT_links_df

 
    for key in ['_trips_i', '_calendar_i', '_calendar_dates_g']:
        if key in feed_dict:
            del feed_dict[key]
    
    return Feed(**feed_dict)

def get_bounding_box(latitude_in_degrees, longitude_in_degrees, half_side_in_m):
    """
    Makes a box around a location using half size of a side and returns the
    mionimum and maximum coordinates in WGS 1984.
    """
    assert half_side_in_m > 0
    assert latitude_in_degrees >= -180.0 and latitude_in_degrees  <= 180.0
    assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0

    half_side_in_km = half_side_in_m / 1000
    lat = math.radians(latitude_in_degrees)
    lon = math.radians(longitude_in_degrees)

    radius  = 6371
    # Radius of the parallel at given latitude
    parallel_radius = radius * math.cos(lat)

    lat_min = lat - half_side_in_km/radius
    lat_max = lat + half_side_in_km/radius
    lon_min = lon - half_side_in_km/parallel_radius
    lon_max = lon + half_side_in_km/parallel_radius
    
    rad2deg = math.degrees

    lat_min = rad2deg(lat_min)
    lon_min = rad2deg(lon_min)
    lat_max = rad2deg(lat_max)
    lon_max = rad2deg(lon_max)

    return {'lat_min':lat_min, 'lat_max':lat_max, 'lon_min':lon_min, 'lon_max':lon_max}

def around_stops(
    stops,                   #GTFS feed df of stops
    walk_duration_sec,       #Walking time
    walk_speed_kmh,          #Walking speed
    lat,                     #Origin Lat
    lon                      #origin Lon
):
    """
    This function gets a stops in pd format and a location. Then extracts stops that
    are around the input stop based on the walk duration and speed.
    Note:
        - stops is a pandas dataframe of stops.txt in GTFS
    """
    assert lat >= -180.0 and lat  <= 180.0
    assert lon >= -180.0 and lon <= 180.0

    walk_dist_m = walk_speed_kmh * 1000 / 3600 * walk_duration_sec
    
    box = get_bounding_box(lat, lon, walk_dist_m)

    cond =  (stops['stop_lat'] > box['lat_min'])&\
            (stops['stop_lon'] > box['lon_min'])&\
            (stops['stop_lat'] < box['lat_max'])&\
            (stops['stop_lon'] < box['lon_max'])

    stops_df = stops[cond].copy()
    if not stops_df.empty:
        cols = ['stop_lat', 'stop_lon']
        stops_df['walk_to_stop_m'] = stops_df[cols].apply(lambda row: hp.distance_m(lat, lon, row[0], row[1]), axis = 1)
        stops_df['walk_to_stop_sec'] = stops_df['walk_to_stop_m'] / walk_speed_kmh / 1000 * 3600
        stops_df = stops_df[['stop_id', 'walk_to_stop_sec']][stops_df['walk_to_stop_sec'] < walk_duration_sec].copy()
        stops_df.reset_index(drop = True)
        
        
    return stops_df

def connect2ptnetwork(
    pt_network,
    start_location, #(x, y)
    walk2transit):
    """
    Generates a dataframe connecting all nodes (stops and origins) in the network to each other:
    The connection is based on walking time and walking speed.
    """
    feed = pt_network.feed
    stops = feed.stops
    
    origin_lat = start_location[1]
    origin_lon = start_location[0]
    walk_graph = around_stops(
        stops, 
        walk2transit, 
        pt_network.walk_speed_kph, 
        origin_lat,
        origin_lon)
    walk_graph['o_type'] = 'Start Location'
    walk_graph.rename(columns = {"stop_id": "d_stop"}, inplace = True)
    return walk_graph

def get_slice(links_df, around_stops_df):
    """
    the objective here is to ge a list of stops and find out what trips pass these stops after
    a specific time. Only trips that leave after we arrive at stop are valid. The last trip we can 
    take is the one leaves before the total time we are willing to wait.
    Notes:
       - links_df is cleand version of all links in GTFS. By clean I mean no links for before our analysis
         starts and no links after our analysis ends.
       - stop_seq_df is a data frame of trips and sequence of stops
       - around_stops_df is a data frame of stops where people can walk to. It is also cleaned and
         actual arrival time in text and second formats has been added
    First, we cut the links that start from stops we can reach
    Second, we remove any service that leaves before we arrive
    third, we add stop sequence to the data frame, this will be later used to remove
    trips with exact same pattern. In other word no one take the second bus of two exact
    same serivces.
    """
    first_link_df = links_df.merge(around_stops_df, left_on='o_stop', right_on='d_stop', how='inner')

    # Now we remove all trip_ids that pass from our stop. In other words, no one would take a service twice.
    ramining_links_df = links_df[~links_df['trip_id'].isin(first_link_df['trip_id'])].copy().reset_index(drop = True)

    cond = first_link_df['o_time'] >= first_link_df['arrival_time']
    first_link_df = first_link_df[cond].copy().reset_index(drop=True)


    
    if not first_link_df.empty:
        
        #This part of the code is a mistry! but it cleans the first link errors and it is curcial to the rest of the code
        first_link_df = first_link_df.drop_duplicates(['stop_seq'])
        def clean_first_link(group):
            arrival_time_sec = group['arrival_time_sec'].min()
            output = group[group['arrival_time_sec'] == arrival_time_sec].copy()
            output = output.drop_duplicates(['trip_id'])
            return output

        first_link_df = first_link_df.groupby('trip_id').apply(clean_first_link).reset_index(drop = True)
        #=================================================================================================================
        

        first_link_df['wt'] = first_link_df['o_time_sec'] - first_link_df['arrival_time_sec']

        # now we select the links that shape the tail of trips we can reach. By tail I mean
        # the ramining of the a full trip that is after my stop.cmd
        first_link_df = first_link_df[['trip_id', 'o_sequence', 'arrival_time_sec', 'd_tt', 'wt', 'awt']].copy()

        first_link_df.rename(columns = {'o_sequence': 'min_seq',
                                        'arrival_time_sec': 'arrive_at_link',
                                         'd_tt': 'o_tt',
                                        'awt': 'o_awt'}, inplace = True)

        #selects all trips that pass based on trip id from links_df
        first_trips_df = links_df.merge(first_link_df).reset_index(drop = True)

        cond = (first_trips_df['o_sequence'] >= first_trips_df['min_seq'])
        first_trips_df = first_trips_df[cond].copy().reset_index(drop = True)
        #first_trips_df.awt.fillna(first_trips_df.wt, inplace=True)
        
        first_trips_df['tt'] = (first_trips_df['d_time_sec'] - \
                                first_trips_df['arrive_at_link'] -\
                                first_trips_df['wt'] +\
                                first_trips_df['o_awt'])
        first_trips_df['d_tt'] = first_trips_df['o_tt'] + first_trips_df['tt']
        first_trips_df = first_trips_df.drop(['min_seq'], axis=1)
    else:
        first_trips_df = pd.DataFrame()
    
    return first_trips_df, ramining_links_df

def walk_to_next_stop(walk_edges, previous_slice_df):
    """
    stops_df is from GTFS
    stops_ids is a pandas series
    """
    def add_around_stops(group):
        o_stop = group.name
        o_tt = group['d_tt'].min()
        o_time_sec = group[group['d_tt'] == o_tt]['d_time_sec'].min() #end of the link is start of walk to next stop
        around_stops_df = walk_edges[walk_edges['o_stop'].isin(previous_slice_df['d_stop'])].copy()

        around_stops_df.rename(columns = {'stop_id': 'd_stop'}, inplace = True)        
        around_stops_df['arrival_time_sec'] = around_stops_df['walk_to_stop_sec'] + o_time_sec
        around_stops_df['arrival_time'] = around_stops_df['arrival_time_sec'].map(hp.sec2text)
        around_stops_df['o_tt'] = o_tt
        return around_stops_df
        
    around_stops_df = previous_slice_df.groupby('d_stop').apply(add_around_stops).reset_index(drop = True)
    
    around_stops_df['d_tt'] = around_stops_df['o_tt'] + around_stops_df['walk_to_stop_sec']
    
    around_stops_df = around_stops_df.sort_values(['d_stop', 'd_tt'])
    around_stops_df = around_stops_df.groupby('d_stop').first().reset_index()
    #around_stops_df.drop('d_tt', axis=1, inplace=True)
   
    return around_stops_df

def build_pt_network(
    feed,
    analysis_start = '07:00:00',    #time in string 'HH:MM:SS'    
    analysis_end = '09:00:00',      #time in string 'HH:MM:SS'
    transfer_duration = 2,           #2 minutes
    walk_speed_kph = 4.8,           #Walking speed in kilometer per hour
    convert_to_gpd = False,         #generates a geopandas database
):
    
    assert gt.valid_time(analysis_start)
    assert gt.valid_time(analysis_end)
    
    pt_net = pt_network(
        feed=feed, 
        analysis_start=analysis_start, 
        analysis_end=analysis_end,
        transfer_duration=transfer_duration, 
        walk_speed_kph=walk_speed_kph)
    
    
    #calculates the average wait time (awt) depending on the analysis awt period.
    PT_links_df = feed.feed_segments
    #removes the PT links outside the analysis awt period
    cond = (PT_links_df['o_time_sec'] >= pt_net.analysis_start_sec)&\
           (PT_links_df['d_time_sec'] <= pt_net.analysis_end_sec)
    PT_links_df = feed.feed_segments[cond].copy()

    #calculates the frequency of trips
    frq_df = PT_links_df['stop_seq'].value_counts().reset_index()
    frq_df.columns = ['stop_seq', 'service_cnt']
    frq_df['headway_sec'] = (pt_net.analysis_duration_sec) /frq_df['service_cnt'] 
    frq_df['headway_min'] = frq_df['headway_sec'] / 60
    PT_links_df = PT_links_df.merge(frq_df, how = 'left')

    #calculates the awt
    PT_links_df['awt'] = PT_links_df['headway_sec'] / 2 #average waite time (sec) is half the headway

    if convert_to_gpd == True:
        #converting the PT_links_df to a geodataframe
        l = lambda x: geom.LineString([geom.Point(x.o_stop_lon,x.o_stop_lat), geom.Point(x.d_stop_lon, x.d_stop_lat)])
        PT_links_df['geometry'] = PT_links_df.apply(l, axis=1)
        PT_links_gdf = gp.GeoDataFrame(PT_links_df)
        pt_net.ptedge = PT_links_gdf
    else:
        pt_net.ptedge = PT_links_df
        
    #connecting stops together with direct walking
    stops = feed.stops
    
    walk_graph = list()
    for stop in stops[['stop_id', 'stop_lat', 'stop_lon']].iterrows():
        s = around_stops(
            stops, 
            pt_net.transfer_duration_sec, 
            walk_speed_kph, 
            stop[1][1],
            stop[1][2])
        s['o_stop'] = stop[1][0]
        s['o_type'] = 'GTFS stop'
        walk_graph.append(s)
    wedge = pd.concat(walk_graph)
    wedge.rename(columns = {"stop_id": "d_stop"}, inplace = True)
    
    pt_net.wedge = wedge

    return pt_net

def abundant_access_single(
    pt_network,
    start_location,        #(x, y)
    transfers = 999,       #number of transfers
    walk_to_transit = 5,   # minutes
    walk_from_transit = 5,   #minutes
    ): 
    """
    The objective here is to find how much of a city is available to by PT from some locations at an specific time.
        - The analysis date has to be valid for the feed. The format should be like 20170101
        - The analysis time is text like '08:00:00'
    The output can be point or polygon. If point, each point feature will have an attribute showing the remaining time
    for walking from that point. If polygon, the output will be one multipart feature showing all 
    
    """
    #connect the start location to the pt network
    feed = pt_network.feed
    ptedge = pt_network.ptedge
    wedge = pt_network.wedge
    
    walk_to_transit = 5 * 60      #seconds
    walk_from_transit = 5 * 60    #seconds
    
    #finding around stops
    ar_df = connect2ptnetwork(
        pt_network,
        start_location, #(x, y)
        walk_to_transit)
    ar_df['arrival_time_sec'] = pt_network.analysis_time_sec + ar_df['walk_to_stop_sec']
    ar_df['arrival_time'] = ar_df['arrival_time_sec'].apply(hp.sec2text)
    ar_df['o_tt'] = 0
    ar_df['tt'] = ar_df['walk_to_stop_sec']
    ar_df['d_tt'] = ar_df['o_tt']+ ar_df['walk_to_stop_sec']
    
    #calculating abundant access
    rl_df = ptedge # rl is remainging links
    at = []
    transfer = 0     
    while True:
        #ft is first tirps
        ft_df, rl_df = get_slice(rl_df, ar_df)
        ft_df['transfer'] = transfer
        transfer += 1
        at.append(ft_df)
        if (ft_df.empty) or (transfer >= transfers): 
            break
        ar_df = walk_to_next_stop(wedge, ft_df)

    all_trips = pd.concat(at).reset_index(drop = True)
    if all_trips.empty:
        return pd.DataFrame()

    #converting lines to points
    def iso_p(group):
        stop_arrive_sec = group['d_time_sec'].min()
        awt_travel_time = group['d_tt'].min()
        stop_id = group['d_stop'].iat[0]
        stop_lat = group['d_stop_lat'].iat[0]
        stop_lon = group['d_stop_lon'].iat[0]
        buffer_dict = {'stop_arrive_sec': stop_arrive_sec,
                       'awt_travel_time': awt_travel_time,
                       'stop_id': stop_id,
                       'stop_lat': stop_lat,
                       'stop_lon': stop_lon}
        return pd.DataFrame(buffer_dict, index=[0])

    buffer_df = all_trips.groupby(['d_stop']).apply(iso_p).reset_index(drop = True)
    return buffer_df

def abundant_access(
    pt_network,                 
    start_locations,             #a gdf of points where you want to calculate catchments for.
    transfers = 999,       #number of transfers
    walk_to_transit = 5 * 60,
    walk_from_transit = 5 * 60,
    cut_off_times = [30],        #in minutes
    polygon = False):
    
    """
    The objective here is to find how much of a city is available to by PT from some locations.
    The analysis period usually refers to AM peak, inter peak or pm peak but any period can be defined here.
    Please note:
        - feed is a gtfstk feed object
        - locations is a data frame with columns [id, lat, lon, values..]
        - two format are accepted 1)fullwd eg Thursday 2)date in YYYYMMDD eg 20170613
        - rings around the catchments for different travel times
        - number of transfers
        - walk to transit
        - 
    
    """
    #all time inputs should be in seconds
    cut_off_times = [x*60 for x in cut_off_times]    #convert minutes to seconds

    #inferred inputs also in seconds
    max_cuf_off = max(cut_off_times)
    assert max_cuf_off < pt_network.analysis_duration_sec/2

   
    aas_list = list()   #aas stands for all accessible stops
    start_locations.reset_index(inplace = True)
    for start_location in start_locations[['geometry']].itertuples():
        location = (
            start_location.geometry.x,
            start_location.geometry.y,
        )

        #as is accessible stops
        as_df = abundant_access_single(
            pt_network,
            location,        #(x, y)
            transfers,       #number of transfers
            walk_to_transit,
            walk_from_transit,
            )
        as_df['o_stop']= start_location.Index
        aas_list.append(as_df)
        
        print('location number {0} has been calculated'.format(start_location.Index))
        
    aas_df = pd.concat(aas_list)
    
    aas_df['geometry'] = aas_df.apply(lambda row: geom.Point(row.stop_lon,row.stop_lat), axis=1)
    aas_df = gp.GeoDataFrame(aas_df, crs = {'init' :'epsg:4326'})
    
    
    
    if polygon:
        aas_df = hp.to_utm(aas_df)
        buffers = list()
        for cut_off_time in cut_off_times:
            aas_df[cut_off_time] = ((cut_off_time - aas_df['awt_travel_time'])/3600)*5000
            aas_df[cut_off_time] = aas_df[cut_off_time].map(lambda x: x if x < walk_from_transit else walk_from_transit)
            b = aas_df[aas_df[cut_off_time] > 0].copy()          
            b['geometry'] = b.apply(lambda row: row.geometry.buffer(row[cut_off_time]), axis=1)
            b['dissolve'] = 'buffer {0}'.format(cut_off_time)
            buffers.append(b.dissolve(by='dissolve'))
        buffers_df = pd.concat(buffers).reset_index()
        buffers_gdf = gp.GeoDataFrame(buffers_df)
        return buffers_gdf[['dissolve', 'geometry']].copy()
    else:
        return aas_df 