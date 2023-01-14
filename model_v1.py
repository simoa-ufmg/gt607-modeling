# warnings
import warnings
# time
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

# geospatial analysis
# import ee
import collections
import geopandas as gpd


# date manipulation
from datetime import timedelta
from datetime import datetime


# windows folder manipulation
import os
import glob

# statistics
import statistics
from statistics import mean
import scipy
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.stats import zscore

# regressions, metrics and models optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
# split database in training and testing data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # randomforest regression
# linear regression or multiple linear regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor  # neural network
from sklearn.svm import SVR  # svm regression
from sklearn.model_selection import cross_validate, GridSearchCV, RepeatedKFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, make_scorer
from xgboost import XGBRegressor

# dataframe storage
import pickle

import model.model_class as model_class

warnings.simplefilter('ignore')

region = "TM"


def func(df, model, transform, wqp):
    # importa os atributos de entrada
    # input_attributes = model.feature_names_in_.tolist()

    # print(f"data: {data.shape} | x: {bands.shape} ")

    # transforma matriz para vetor para realizar operacao
    # stacked = xarray.stack(z=("x", "y"))
    # return stacked
    # seleciona as bandas
    # stacked2[[c for c in xarray.columns if c in loaded_model.feature_names_in_.tolist()]]
    # stacked2 = xarray.sel(bands=loaded_model.feature_names_in_.tolist())
    # print(stacked2.T.shape)
    # print(stacked2.T.values)
    # print(stacked2.T.max())
    # df_temp = stacked2.T.to_dataframe(name = "SIMOA")
    # # return df_temp
    y_pred = model.predict(df.values)  # predicao do modelo
    return y_pred
    # #y_pred_da = xr.DataArray(np.reshape(np.exp(y_pred), (y_pred.shape[0], 1)), dims=("z", "bands"))
    # if transform == 'exp':
    #     y_pred_da = xr.DataArray(np.exp(y_pred), dims=("z"))
    # else:
    #     y_pred_da = xr.DataArray(y_pred, dims=("z"))

    # y_pred_da["bands"] = wqp
    # #y_pred_da.assign_coords({"bands": "Chla"})
    # temp = xr.concat(objs=[stacked.T, y_pred_da], dim='bands')

    # temp = temp.unstack("z")
    # return temp


model_dir = "./best_models/TM_30m"

model_names = [x.split(
    "_")[0] + "_model" for x in os.listdir(model_dir) if not x.startswith(".ipyn")]
model_files = [model_dir + "/" +
               model for model in os.listdir(model_dir) if not model.startswith(".ipyn")]

model_dict = {idx: value for (idx, value) in zip(model_names, model_files)}
data_dir = glob.glob("*-TM")
data_files = []
for i in data_dir:
    data = glob.glob(os.path.join(i, "chl_algo_median_reflectance_30.csv"))
    data_files.append(data)

metadata_files = []
for j in data_dir:
    metadata = glob.glob(os.path.join(j, "metadata.csv"))
    metadata_files.append(metadata)

metadata_df = []
for meta in metadata_files:
    df = pd.read_csv(meta[-1], index_col=None, header=0)
    metadata_df.append(df)


li = []

k = 0
for filename in data_files:
    df = pd.read_csv(filename[-1], index_col=None, header=0)
    df = pd.concat([df, metadata_df[k]], axis=1)
    k += 1
    li.append(df)

# df = df.drop(columns=['Image'])
df = pd.concat(li, axis=0, ignore_index=True)
df = df.drop(columns=['Gimbal Degree'])
if region == "Pampulha":
    df = df.drop(columns=['Degree'])
df = df.drop(columns=['Altitude'])
# df = df.drop(columns=['R/B'])
# df = df.drop(columns=['R^2/B'])
# df = df.drop(columns=['MLR'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(0)
df.dropna(inplace=True)
df['R^2/NIR'] = pd.to_numeric(
    df['R^2/NIR'], downcast='float')
df['R^2/RE'] = pd.to_numeric(
    df['R^2/RE'], downcast='float')
df['R^2/B'] = pd.to_numeric(
    df['R^2/B'], downcast='float')
df['MLR'] = pd.to_numeric(
    df['MLR'], downcast='float')

model_selected = pd.read_csv("./best_models/TM_30m/model_selected.csv")
model_dict.pop("model_model")
x = 0
for i in model_dict:
    df_to_predict = df
    x += 1

    filename = model_dict[i]
    loaded_model = pickle.load(open(filename, 'rb'))
    print(filename)
    model_name = filename.split('/')
    model_name = model_name[-1].split('_')
    model_linear = model_name[2]
    if model_linear == "LinearRegression":
        df_to_predict = df_to_predict["NDVI"]
        df_to_predict = df_to_predict.to_frame(name="NDVI")

    if i == 'secchi_model':
        wqp = 'Secchi (cm)'
        transform = 'linear'
    elif i == 'fdom_model':
        wqp = 'Fdom (fnu)'
        transform = 'linear'
    elif i == 'pc_model':
        wqp = 'PC (rfu)'
        transform = 'linear'
    elif i == 'chl_model':
        wqp = 'Chl (rfu)'
        transform = 'log'
    elif i == 'turb_model':
        wqp = 'Turb (fnu)'
        transform = 'linear'
    if transform == 'log':
        df_to_predict = df_to_predict.apply(lambda x: np.log(
            x) if np.issubdtype(x.dtype, np.number) else x)
    elif transform == 'poly':
        poly = PolynomialFeatures(degree=2)
        df_to_predict = poly.fit_transform(df_to_predict.values)
    elif transform == 'exp':
        df_to_predict = df_to_predict.apply(lambda x: np.exp(
            x) if np.issubdtype(x.dtype, np.number) else x)
    if region == "TM":
        if i != 'fdom_model' and i != 'turb_model':
            df_to_predict = df_to_predict[[
                c for c in df_to_predict.columns if c in loaded_model.feature_names_in_.tolist()]]
    elif region == "Pampulha":
        if i != "secchi_model":
            df_to_predict = df_to_predict[[
                c for c in df_to_predict.columns if c in loaded_model.feature_names_in_.tolist()]]
    elif i == 'fdom_model' or i == 'chl_model':
        df_to_predict['R^2/NIR'] = pd.to_numeric(
            df_to_predict['R^2/NIR'], downcast='float')
        df_to_predict['R^2/RE'] = pd.to_numeric(
            df_to_predict['R^2/RE'], downcast='float')
        df_to_predict['R^2/B'] = pd.to_numeric(
            df_to_predict['R^2/B'], downcast='float')
        df_to_predict['MLR'] = pd.to_numeric(
            df_to_predict['MLR'], downcast='float')
    print(i)
    print(model_name[2])
    indices_to_keep = ~df_to_predict.isin([np.nan, np.inf, -np.inf]).any(1)
    df_to_predict = df_to_predict[indices_to_keep]
    print(df_to_predict.columns)
    df_to_predict = df_to_predict.reset_index()
    predict_df = func(df_to_predict, loaded_model, 'exp', wqp)
    df[wqp + ' ' + model_name[2]] = predict_df
# df.to_csv("TM_models_30M_0.3.csv")


# for i in data_dir:
#     file_path = os.path.join(i, "chl_algo_median_reflectance_30.csv")
#     df.to_csv(file_path)

# filename = model_dict.get('secchi_model')
# loaded_model = pickle.load(open(filename, 'rb'))

# print(loaded_model.feature_names_in_)
