""" Copyright (c) University of Chicago - All Rights Reserved
 You may use, distribute and modify this code under the
 terms of the XYZ license.
 
 You should have received a copy of the XYZ license with
 this file. If not, please write to: , or visit :

metric.py contains all of the primitives for defining sociome metrics.

A metric is a spatial function that estimates an SDOH metric at a particular
latitude an longitude point. For example, one can calculate the distance
to any park in a city. This is a function over all latitude and longitude
pairs
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from geopy import distance

from sklearn.neighbors import BallTree
from rtree import index


# defines the main super class for all of the spatial functions
# a spatial function is a set of points on a map associated with a given value
class SpatialFunction(object):
    '''A spatial function assigns values to every row in a geopandas dataframe
	'''

    # takes a geodataframe of point geometry and a metric column
    def __init__(self, gdf, metric_col=None):

        self.gdf = gdf[~gdf['geometry'].is_empty]

        self.X = np.array([self.gdf['geometry'].x.to_numpy(), \
                           self.gdf['geometry'].y.to_numpy()]).T

        self.N = self.X.shape[0]

        if metric_col is None:
            self.FX = np.zeros(self.N)
        else:
            self.FX = self.gdf[metric_col].to_numpy()

        self.metric_col = metric_col

    def query(self, x):
        return np.mean(self.FX[np.all(self.X == x, axis=1)])

    def to_gdf(self, pts):
        N = pts.shape[0]
        data = []
        for i in range(N):
            fx = self.query(pts[i, :].reshape(1, -1))
            data.append({'x': pts[i, 0], 'y': pts[i, 1], 'fx': fx})

        df = pd.DataFrame(data)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    def augment(self, gdf, as_name):
        pts = np.array([gdf['geometry'].x.to_numpy(), \
                        gdf['geometry'].y.to_numpy()]).T
        output = self.to_gdf(pts)
        output[as_name] = output['fx']

        return gdf.merge(output, on='geometry')

    def print(self):
        return print(self.gdf[[self.metric_col, 'geometry']].head())


# assigns a distance to a certain set of points
class SpatialVoronoiFunction(SpatialFunction):
    # a spatial function is a set of points on a map associated with a given value

    # takes a geodataframe of point geometry and a metric column
    def __init__(self, gdf):
        super(SpatialVoronoiFunction, self).__init__(gdf)
        self.tree = BallTree(self.X)

    def query(self, x):
        _, ind = self.tree.query(x, k=1)

        ind = ind.flatten()
        return distance.distance(x, self.X[ind, :]).meters


# assigns a count within a radius for points
class SpatialDensityFunction(SpatialFunction):
    # a spatial function is a set of points on a map associated with a given value

    # takes a geodataframe of point geometry and a metric column
    def __init__(self, gdf, radius=0.01):
        super(SpatialDensityFunction, self).__init__(gdf)
        self.tree = BallTree(self.X)
        self.bandwidth = radius

    def query(self, x):
        ind = self.tree.query_radius(x, r=self.bandwidth)[0]
        if len(ind) == 0:
            return 0

        return len(ind)


# interpolates a continuous function
class SpatialInterpolationFunction(SpatialFunction):
    # a spatial function is a set of points on a map associated with a given value

    # takes a geodataframe of point geometry and a metric column
    def __init__(self, gdf, metric_col, sigma2=8e-3, precision=1e-6):
        super(SpatialInterpolationFunction, self).__init__(gdf, metric_col)
        self.tree = BallTree(self.X)
        self.sigma2 = sigma2
        self.bandwidth = -np.log(precision) * sigma2

    def query(self, x):
        ind = self.tree.query_radius(x, r=self.bandwidth)[0]
        if len(ind) == 0:
            return np.NaN

        norm_list = np.sum(np.power(self.X[ind] - x, 2), axis=1)
        explist = np.exp(-norm_list / self.sigma2)
        return np.dot(explist, self.FX[ind])  # /np.sum(explist)
