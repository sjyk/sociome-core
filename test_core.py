##test routines for the core packages

from city import CityScape
from data import *


city = CityScape.chicago()

city.add_metric('SE_A14006_')
city.add_metric('Distance-to-Park', get_chicago_parks_distance())
city.add_metric('Crime Density', get_chicago_crime())
city.add_metric('Air Quality RPCA',  get_air_quality_rpca_8um6())
city.add_metric('Construction', get_construction())
city.add_metric('Traffic', get_traffic())
city.add_metric('Graffiti', get_graffiti())

city.visualize()

city.to_kepler('map.html', ['Crime Density'])