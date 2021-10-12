# import packages

import sys
sys.path.insert(1, r'D:\Git_repos\Flask-Application\de-smart-monitoring-backend-module-master')
import de_sm_backend.nrt_io.downloader as downloader
from flask import Flask, render_template, request, redirect, jsonify, url_for, send_file, redirect, make_response, Response, send_from_directory,after_this_request
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import box
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import json
import io
import os
import random
from PIL import Image
import base64
import re
import xarray
import zipfile

# Import class

from layer import Layer
from layer import create_roi

# declare name and location of files for the application
app = Flask(__name__,
            static_folder="static",
            template_folder="templates")

app.config["DOWNLOAD_NC"] = os.path.join(app.static_folder,"download")

dataset_a = {"path":r"D:\Umweltnaturwissenschaften\7. Semester\Bachelor\Daten\Study domain\germany_boundaries\DEU_adm1.shp",
             "data_type":"shapefile",
             "type":"local"}

dataset_b = {"path":"",
             "data_type":"raster",
             "download_func":"weatherprediction()",
             "type":"download"}

dataset_c = {"path":r"D:\Umweltnaturwissenschaften\7. Semester\Bachelor\Daten\Study domain\Jasper\dem_study_domain.tif",
             "data_type":"raster",
             "type":"local"}

default = {
            "calc_operator": "false",
            "filter_operator": "false",
            "boundaries": "false",
            "palette": "viridis",
            "extract": "none",
            "marker_loc": "false",
            "time":"false"
           }

# render 'entrance' website
@app.route("/")
def home():
    return render_template("home.html")

# render application website
@app.route("/map")
def map():
    plt = json.dumps(go.Figure([go.Scatter(x=[], y=[])]).update_layout(template="plotly_dark"),
                     cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("map.html",plt=plt)

# render website for application with get and post methods
@app.route("/map/data",methods=["GET","POST"])
def map_data():

    dataset = request.args["dataset"]
    print(request.args)

    if dataset == "":

        default["error"] = "No dataset was selected"
        default["variable"] = request.args["variable"]
        py_data = default

        return py_data


    js_data = {
        "calc_operator": request.args["calc_operator"],
        "filter_operator": request.args["filter_operator"],
        "boundaries": request.args["boundaries"],
        "palette": request.args["palette"],
        "extract": request.args["value_extraction"],
        "marker_loc": request.args["marker_loc"],
        "variable": request.args["variable"],
        "time": request.args["time"],
        "animation": request.args["animation"],
        "download": request.args["download"]
    }

    if eval(dataset)["type"]=="download":
        eval(dataset)["path"] = xarr_dataset


    kwargs = dict(eval(dataset), **js_data)

    layer = Layer(**kwargs)
    py_data = layer.get_json()

    return py_data

# return json of possible variables of data set
@app.route("/map/data/variable",methods=["POST","GET"])
def map_variable():
    selected = request.args["selected"]
    boundaries = request.args["boundaries"]

    kwargs = eval(selected)


    error = "false"
    time_cnt = []
    variables = []
    xarr = kwargs["type"] == "download"

    if kwargs["data_type"]=="shapefile":
        geo_file = gpd.read_file(kwargs["path"])
        geo_file["prec"] = [random.randint(0, 10) for x in range(0, geo_file.shape[0])]
        geo_file["temp"] = [random.randint(15, 30) for x in range(0, geo_file.shape[0])]
        variables_temp = geo_file.select_dtypes(include="number").columns.values.tolist()
        variables = [i.lower() for i in variables_temp if not "id" in i.lower()]
    else:
        if xarr:
            if boundaries=="false":
                error = "A Polygon or rectangle needs to be drawn as boundary to load the dataset"
            else:
                global xarr_dataset
                xarr_dataset = eval("downloader.downloader(roi=create_roi(boundaries))."+kwargs["download_func"])
                variables = list(xarr_dataset.keys())
                try:
                    time_cnt = {f"{str(pd.Timestamp(key).year)}-{str(pd.Timestamp(key).month)}-{str(pd.Timestamp(key).day)} {str(pd.Timestamp(key).hour)}:00":value for (key,value) in zip(xarr_dataset.coords["time"].values,
                                                                                                                                                                   [i for i in range(0,len(xarr_dataset.coords["time"].values))])}
                except:
                    time_cnt = []

        else:
            file_path = kwargs["path"]
            with rasterio.open(file_path) as src:
                band_cnt = src.meta["count"]
                variables = [f"band {i + 1}" for i in range(0, band_cnt)]

    return {"variables":variables,"error":error,"xarr":xarr,"time":time_cnt}


# return plt json
@app.route("/map/plot/data",methods=["POST","GET"])
def plot_data():

    dataset = request.args["inset_dataset"]

    if dataset == "":
        return json.dumps(go.Figure([go.Scatter(x=[], y=[])]).update_layout(template="plotly_dark"),
                     cls=plotly.utils.PlotlyJSONEncoder)

    default["variable"] = request.args["inset_variable"]

    kwargs = dict(eval(dataset), **default)
    graph = Layer(**kwargs)
    py_data = graph.get_graphJSON()


    return py_data

#download xarray
@app.route("/map/download_prep",methods=["GET"])
def download_prep():

    print(request.args)

    data_dict = json.loads(request.args["data"])
    global file_numb
    file_numb = len(data_dict)

    if file_numb==1:

        dataset = data_dict[list(data_dict.keys())[0]]
        dataset_name = dataset["dataset"]

        js_data = {
            "calc_operator": dataset["calc_operator"],
            "filter_operator": dataset["filter_operator"],
            "boundaries": dataset["boundaries"],
            "palette": dataset["palette"],
            "extract": dataset["value_extraction"],
            "marker_loc": dataset["marker_loc"],
            "variable": dataset["variable"],
            "time": dataset["time"],
            "animation": dataset["animation"],
            "download": dataset["download"]
        }

        kwargs = dict(eval(dataset_name), **js_data)
        kwargs["path"] = xarr_dataset

        layer = Layer(**kwargs)

        file = layer.get_xarr()

        xarr = xarray.open_rasterio(file)
        xarr.to_netcdf(os.path.join(app.config["DOWNLOAD_NC"] ,f"{dataset['name']}.nc"))

    else:
        for key,item in data_dict.items():
            dataset = data_dict[key]
            dataset_name = dataset["dataset"]

            js_data = {
                "calc_operator": dataset["calc_operator"],
                "filter_operator": dataset["filter_operator"],
                "boundaries": dataset["boundaries"],
                "palette": dataset["palette"],
                "extract": dataset["value_extraction"],
                "marker_loc": dataset["marker_loc"],
                "variable": dataset["variable"],
                "time": dataset["time"],
                "animation": dataset["animation"],
                "download": dataset["download"]
            }

            kwargs = dict(eval(dataset_name), **js_data)
            kwargs["path"] = xarr_dataset

            layer = Layer(**kwargs)

            file = layer.get_xarr()

            xarr = xarray.open_rasterio(file)
            xarr.to_netcdf(os.path.join(app.config["DOWNLOAD_NC"], f"{dataset['name']}.nc"))

    return ('', 204)


@app.route("/map/download")
def download():

    if file_numb==1:

        filename = os.listdir(app.config["DOWNLOAD_NC"])[0]
        file_path = os.path.join(app.config["DOWNLOAD_NC"],filename)


        return_data = io.BytesIO()
        with open(file_path, 'rb') as fo:
            return_data.write(fo.read())
        return_data.seek(0)

        os.remove(file_path)

        return send_file(return_data, as_attachment=True, attachment_filename=filename,mimetype="application/netcdf")

    else:

        files = os.listdir(app.config["DOWNLOAD_NC"])
        file_paths = [os.path.join(app.config["DOWNLOAD_NC"],i) for i in files]

        return_data = {}

        for count,i in enumerate(file_paths):

            return_data[files[count]] = io.BytesIO()
            with open(i, 'rb') as fo:
                return_data[files[count]].write(fo.read())

            return_data[files[count]].seek(0)
            os.remove(i)


        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'a',zipfile.ZIP_DEFLATED,False) as zf:
            for key,item in return_data.items():
                data = zipfile.ZipInfo(key)
                data.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(data,item.getvalue())
        memory_file.seek(0)

        return send_file(memory_file, as_attachment=True, attachment_filename="datasets.zip", mimetype="zip")

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True,use_reloader=False)
