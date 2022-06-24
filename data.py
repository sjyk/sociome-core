""" Copyright (c) University of Chicago - All Rights Reserved
 You may use, distribute and modify this code under the
 terms of the XYZ license.
 
 You should have received a copy of the XYZ license with
 this file. If not, please write to: , or visit :

data.py contains the hooks for fetching data
"""
import pandas as pd
import geopandas as gpd
import json

from metrics import *


def get_chicago_parks_distance():
	url = 'https://data.cityofchicago.org/resource/2eaw-bdhe.json'
	parks = pd.read_json(url)

	def parse(x, key):
	    try:
	        return x[key]
	    except:
	        return None

	parks['latitude'] = parks.location.apply(lambda x: parse(x,'latitude'))
	parks['longitude'] = parks.location.apply(lambda x: parse(x,'longitude'))
	parks = parks.dropna(subset=['latitude', 'longitude'])

	gdf = gpd.GeoDataFrame(parks, geometry=gpd.points_from_xy(parks.longitude, parks.latitude))
	return SpatialVoronoiFunction(gdf)


def get_chicago_crime():
	url = "https://data.cityofchicago.org/resource/dfnk-7re6.json"
	crime = pd.read_json(url)

	gdf = gpd.GeoDataFrame(crime, geometry=gpd.points_from_xy(crime.longitude, crime.latitude))
	return SpatialDensityFunction(gdf)


def get_air_quality_rpca_8um6():
	url_air_quality = 'https://data.cityofchicago.org/resource/i9rk-duva.json'
	air_quality = pd.read_json(url_air_quality)

	gdf = gpd.GeoDataFrame(air_quality , geometry=gpd.points_from_xy(air_quality.longitude, air_quality.latitude))
	return SpatialInterpolationFunction(gdf, ':@computed_region_rpca_8um6')

def get_construction():
	building_permit_api = 'https://data.cityofchicago.org/resource/building-permits.json'
	building_permit = pd.read_json(building_permit_api)
	filter = (building_permit['permit_type'] == 'PERMIT - RENOVATION/ALTERATION') | \
			 (building_permit['permit_type'] == 'PERMIT - NEW CONSTRUCTION') | \
			 (building_permit['permit_type'] == 'PERMIT - WRECKING/DEMOLITION')
	building_permit = building_permit[filter]
	gdf = gpd.GeoDataFrame(building_permit, geometry=gpd.points_from_xy(building_permit.longitude, building_permit.latitude))
	return SpatialDensityFunction(gdf)


def get_traffic():
	url = "https://data.cityofchicago.org/resource/pf56-35rv.json"
	traffic = pd.read_json(url)

	gdf = gpd.GeoDataFrame(traffic, geometry=gpd.points_from_xy(traffic.longitude, traffic.latitude))
	return SpatialInterpolationFunction(gdf, 'total_passing_vehicle_volume')


def get_graffiti():
	graffiti_api = 'https://data.cityofchicago.org/resource/8tus-apua.json'
	graffiti = pd.read_json(graffiti_api)

	gdf = gpd.GeoDataFrame(graffiti, geometry=gpd.points_from_xy(graffiti.longitude, graffiti.latitude))
	return SpatialDensityFunction(gdf)


