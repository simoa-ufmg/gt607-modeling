#warnings
import warnings
#time
import time
# data manipulation
import json
import numpy as np
import pandas as pd
import xarray as xr

# geospatial viz
# import folium
# from folium import Choropleth, Circle, Marker, Icon, Popup, FeatureGroup
# from folium.plugins import HeatMap, FastMarkerCluster, MiniMap
# from folium.features import GeoJsonPopup, GeoJsonTooltip

# plot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

#geospatial analysis
#import ee
import collections
import geopandas as gpd


#date manipulation
from datetime import timedelta
from datetime import datetime


#windows folder manipulation
import os, glob

#statistics
import statistics
from statistics import mean
import scipy
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.stats import zscore

#regressions, metrics and models optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split #split database in training and testing data
from sklearn.ensemble import RandomForestRegressor #randomforest regression
from sklearn.linear_model import LinearRegression, Lasso #linear regression or multiple linear regression
from sklearn.neural_network import MLPRegressor #neural network
from sklearn.svm import SVR #svm regression
from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedKFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, make_scorer
from xgboost import XGBRegressor

#dataframe storage
import pickle

import model.model_class as model_class

warnings.simplefilter('ignore')
    
model_dir = "./best_models/TM_30m"

model_names = [x.split("_")[0] + "_model" for x in os.listdir(model_dir) if not x.startswith(".ipyn")]
model_files = [model_dir + "/" +  model for model in os.listdir(model_dir) if not model.startswith(".ipyn")]

model_dict = {idx: value for (idx, value) in zip(model_names, model_files)}
data_dir = glob.glob()


# data = xr.DataArray(im_aligned, dims=("x", "y", "bands"), coords={"bands": ['Blue', 'Green', 'Red', 'Near-IR', 'RedEdge', 'Thermal']})