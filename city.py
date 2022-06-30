""" Copyright (c) University of Chicago - All Rights Reserved
 You may use, distribute and modify this code under the
 terms of the XYZ license.
 
 You should have received a copy of the XYZ license with
 this file. If not, please write to: , or visit :

city.py contains all of the address informatiton for a particular city.
This should be modified to extend to other cities.
"""

import os
import geopandas
import geopandas as gpd
import pandas as pd
import numpy as np
import rtree

# points in the city with addresses
class CityScape(object):
    '''A cityscape is an object that lists key points in a city. These can be landmarks,
       representative points, or even all the addresses in a city.
    '''

    # by default will drop errors
    def __init__(self, df=None, address_col=None, latitude_col=None, longitude_col=None):

        self.acs = False
        self.metrics = []

        # create a blank dataframe
        if df is None:
            df = pd.DataFrame([{}])
            self.gdf = gpd.GeoDataFrame(df)
            return

        # cleaning
        df[latitude_col] = df[latitude_col].where(df[latitude_col].abs() <= 90)
        df[longitude_col] = df[longitude_col].where(df[longitude_col].abs() <= 90)
        df = df.dropna(subset=[latitude_col, longitude_col])

        self.address_col = address_col
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

        self.gdf = gpd.GeoDataFrame(df[[address_col, latitude_col, longitude_col]], \
                                    geometry=gpd.points_from_xy(df[longitude_col], \
                                                                df[latitude_col]))

    # enforcing boundaries takes a while
    def set_boundaries(self, shapefile, enforce=False):
        self.limits = gpd.read_file(shapefile)
        boundary_poly = self.limits.loc[0, 'geometry']

        if enforce:
            N = len(self.gdf)
            for i, r in self.gdf.iterrows():
                intersection = r['geometry'].intersection(boundary_poly)

                if intersection.is_empty:
                    self.gdf.loc[i, 'geometry'] = None

            self.gdf = self.gdf.dropna(subset=['geometry'])

    # adds a metric to the city
    def add_metric(self, name, metric=None):

        if name not in self.gdf.columns:
            self.metrics.append(name)
            self.gdf = metric.augment(self.gdf, name)
        else:
            self.metrics.append(name)

    # exports all data to a file
    def to_file(self, filename):
        self.gdf.to_file(filename)

    # adds ACS data
    def add_acs(self, acs):
        idx = rtree.index.Index()
        for i, r in acs.iterrows():
            idx.insert(i, r['geometry'].bounds)

        N = len(self.gdf)
        self.gdf['tract'] = 0

        for i, r in self.gdf.iterrows():
            geom = self.gdf.loc[i, 'geometry']
            candidate_tract = [int(id_temp) for id_temp in pd.Series(idx.intersection(geom.bounds))]

            if len(candidate_tract) == 0:
                tract = None
            else:
                tract = acs.iloc[candidate_tract[0]]['GEOID']
                self.gdf.loc[i, 'tract'] = tract

        self.acs = True

        self.gdf.set_crs(epsg=4269, inplace=True)
        local_gdf = geopandas.sjoin(self.gdf, acs, how='left', op='within')

        self.gdf = local_gdf

    # visualizes the data with matplotlib
    def visualize(self, n=-1, metrics=None):
        import matplotlib.pyplot as plt

        visualized_metrics = self.metrics
        # print(visualized_metrics)
        if not metrics is None:
            visualized_metrics = metrics

        plots = len(visualized_metrics) + 1

        # clips to only 4 plots
        if plots > 4:
            plots = 4

        fig, ax = plt.subplots(1, plots, figsize=(20, 10))

        if plots == 1:
            ax = [ax]

        self.limits.plot(ax=ax[0], facecolor="none", edgecolor="green")

        # no sampling
        if n == -1:
            sample = self.gdf
        else:
            sample = self.gdf.sample(n=n)

        if self.acs:
            sample.plot(ax=ax[0], column='tract', markersize=2, cmap='Dark2')
        else:
            sample.plot(ax=ax[0], markersize=2)

        ax[0].set_title('Address Points')

        for i in range(1, plots):
            self.limits.plot(ax=ax[i], facecolor="none", edgecolor="green")
            sample.plot(ax=ax[i], column=visualized_metrics[i - 1], cmap='OrRd', markersize=1)
            ax[i].set_title(visualized_metrics[i - 1])

        plt.show()

    # measures the correlation between metrics
    def corr(self, metric):
        dct = {}
        for other in self.metrics:
            dct[(metric, other)] = np.corrcoef(self.gdf[metric].to_numpy(), \
                                               self.gdf[other].to_numpy())

        return dct

    # '477 WEST DEMING PLACE'
    def get_address(self, addr, metric):
        return self.gdf[self.gdf['ADDRDELIV'] == addr][metric]

    def quantile(self, metric):
        return np.quantile(self.gdf[metric], np.arange(0, 1, 0.1))

    # outputs an interactive map
    def to_kepler(self, output, metrics):
        from keplergl import KeplerGl

        config = {
            'version': 'v1',
            'config': {
                'mapState': {
                    'latitude': 41.7418876,
                    'longitude': -87.9063053,
                    'zoom': 10.0
                }
            }
        }

        map1 = KeplerGl(height=400)
        map1.config = config
        map1.add_data(data=self.gdf[metrics + ['ADDRDELIV', 'Lat', "Long", 'geometry']], name="Addresses")
        map1.save_to_html(file_name=output)

    @classmethod
    def from_file(cls, filename, n=-1):
        city = cls()  # create object

        if n > 0:
            city.gdf = gpd.read_file(filename).sample(n)
        else:
            city.gdf = gpd.read_file(filename, rows=10000)

        columns = city.gdf.columns
        city.address_col = columns[0]
        city.latitude_col = columns[1]
        city.longitude_col = columns[1]

        return city

    '''Chicago specific data
    '''

    # loads the chicago data
    @classmethod
    def chicago(cls, n=-1):
        city = CityScape.from_file('gis_data/chicago.pqt', n)
        city.set_boundaries('gis_data/chicago')
        file = os.listdir("gis_data/acs")
        path = [os.path.join("gis_data/acs", i) for i in file if ".shp" in i]
        gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path], ignore_index=True),
                               crs=gpd.read_file(path[0]).crs)

        city.add_acs(gdf)
        return city

    # resets the address data for chicago
    @classmethod
    def reset_chicago(cls):
        df = pd.read_csv('gis_data/Address_Points.csv')
        df = df[df['geocode_muni'] == 'CHICAGO']
        city = CityScape(df, 'ADDRDELIV', 'Lat', "Long")
        city.to_file('gis_data/chicago.pqt')
