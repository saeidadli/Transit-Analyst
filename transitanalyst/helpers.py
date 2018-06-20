import tempfile
import time 
import shutil
import math
import sys

from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import gtfstk as gt

import geopandas as gp
import shapely.geometry as geom

#time decorator
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' % \
              (method.__name__, te-ts))
        return result

    return timed


# General functions
def distance_m(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - math.cos((lat2 - lat1) * p)/2 +  math.cos(lat1 * p) *  math.cos(lat2 * p) * (1 -  math.cos((lon2 - lon1) * p)) / 2
    return 12742 *  math.asin( math.sqrt(a))* 1000

def sec2text(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

def text2sec(in_time):
    l = in_time.split(':')
    return int(l[0]) * 3600 + int(l[1]) * 60 + int(l[2])

def to_utm(gdf):
    geom = gdf['geometry'].iat[0]
    
    #if Multi geometry only get the first part
    if 'Multi' in geom.type:
        geom = geom[0]

    x, y = geom.xy
    lat = list(y)[0]
    lon = list(x)[0]
    crs = gt.get_utm_crs(lat, lon)
    gdf = gdf.to_crs(crs)
    return gdf

#Basic functions
def feed_obj_to_dict(feed_object):
    """
    converts feed object to a feed dictionary.
    """
    feed_dict = dict()
    for key in gt.FEED_ATTRS:
        feed_dict[key] = getattr(feed_object, key)
    feed_dict['dist_units'] = feed_object.dist_units
    return feed_dict

def validate_feed(feed, in_date):
    """
    The function gets a GTFS feed as a dictionary and a date. 
    It uses removes all trips that are not running in that specific date.
    Note:
        -  in_date is a text and its format is "YYYYMMDD".
    """
    old_feed = feed_obj_to_dict(feed)
    new_feed = dict()
    #converts a text into a date
    t_date = datetime.strptime(in_date, "%Y%m%d")
    #finds the weekday of a date
    weekday = t_date.strftime('%A').lower()

    #makes a copy of the calendar pd and find valid service ids for a specific date 
    calendar = old_feed['calendar'].copy()
    cond = (calendar['start_date'] <= in_date)&\
           (calendar['end_date'] >= in_date)&\
           (calendar[weekday] == 1)
            
    new_feed['calendar'] = calendar[cond].copy()
     
    
    #makes a copy of the trips pd and finds valid trips based on valid service ids
    new_feed['trips'] =  old_feed['trips'][old_feed['trips']['service_id'].isin(new_feed['calendar']['service_id'])].copy()
    
    #makes a copy of the calendar_dates pd and finds valid services based on valid service ids
    if 'calendar_dates' in old_feed:
        new_feed['calendar_dates'] =  old_feed['calendar_dates'][old_feed['calendar_dates']['service_id'].isin(new_feed['calendar']['service_id'])].copy()
    
    #makes a copy of routes pd and finds valid routes based on valid trip ids
    new_feed['routes'] =  old_feed['routes'][old_feed['routes']['route_id'].isin(new_feed['trips']['route_id'])].copy()

    #makes a copy of agency pd and finds valid agencies based on valid routes
    if ('agency_id' in old_feed['agency'].columns) and ('agency_id' in old_feed['routes'].columns):
        new_feed['agency'] = old_feed['agency'][old_feed['agency']['agency_id'].isin(new_feed['routes']['agency_id'])].copy()
    else:
        new_feed['agency'] = old_feed['agency']

    #makes a copy of stop_times pd and finds valid stop_times based on valid trip ids
    new_feed['stop_times']  =  old_feed['stop_times'][old_feed['stop_times']['trip_id'].isin(new_feed['trips']['trip_id'])].copy()

    #makes a copy of stops pd and finds valid stops based on valid stop_ids
    new_feed['stops'] =  old_feed['stops'][old_feed['stops']['stop_id'].isin(new_feed['stop_times']['stop_id'])]

    #makes a copy of shapes pd and finds valid shapes based on valid shape_ids
    if 'shapes' in old_feed:
        new_feed['shapes'] =  old_feed['shapes'][old_feed['shapes']['shape_id'].isin(new_feed['trips']['shape_id'])]
    for tbl in new_feed:
        new_feed[tbl] = new_feed[tbl].reset_index(drop=True)
    
    new_feed['dist_units'] = old_feed['dist_units']
    
    return gt.Feed(**new_feed)