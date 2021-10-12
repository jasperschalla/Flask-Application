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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import io
import random
from PIL import Image
import base64
import re
import xarray
import tempfile
import shutil
from moviepy.editor import ImageSequenceClip


class Layer:

    def __init__(self,**kwargs):
        random.seed(30)


        self.geo_file_shp = "false"
        self.geo_file_rast = "false"
        self.error = "false"
        self.raster_bounds = "false"
        self.raster_range = "false"
        self.value = "false"
        self.raster_stats = "false"
        self.time = "false"

        self.path = kwargs.get("path")
        self.data_type = kwargs.get("data_type")
        self.calc_operator = kwargs.get("calc_operator")
        self.filter_operator_temp = kwargs.get("filter_operator")
        self.boundaries = kwargs.get("boundaries")
        self.palette = kwargs.get("palette")
        self.extract = kwargs.get("extract")
        self.marker_loc = json.loads(kwargs.get("marker_loc"))
        self.variable = kwargs.get("variable")
        self.time_cnt = kwargs.get("time")
        self.animation = kwargs.get("animation")
        self.type = kwargs.get("type")
        self.download = kwargs.get("download")


        xarr = self.type=="download"


        if self.data_type=="shapefile":

            # Read dataset
            self.file = gpd.read_file(self.path)
            self.file["prec"] = [random.randint(0, 10) for x in range(0, self.file.shape[0])]

            # Check if the crs of the shp is 4326 and reproject when not
            self.check_crs_shp()

            # Check if shp needs to be clipped and clip it
            self.clip_shp()

            # Check if filter is applied and apply filter
            self.apply_filter_shp()

            # Check if values of the shp are altered and apply it
            self.calc_shp()

            # Check if values are extracted of the shp and apply it
            self.extract_shp()

            self.file.columns = self.file.columns.str.lower()
            self.geo_file_shp = self.file.to_json()

        else:

            if xarr:
                band = 1
                self.dirpath = tempfile.mkdtemp()
                self.set_up_xarray()

            else:
                band = int(re.compile("(\d+)").search(self.variable).group(1))

            pal = self.pal_rast()
            pal.set_bad(color="grey",alpha=0)


            if xarr and self.animation=="true" and self.time != "false":

                count = len(self.animation_rds.time)

                minimum = self.animation_rds[self.variable].min().values
                maximum = self.animation_rds[self.variable].max().values
                stack_lst = []

                norm = matplotlib.colors.Normalize(vmin=minimum.item(), vmax=maximum.item(), clip=True)
                mapper = cm.ScalarMappable(norm=norm, cmap=pal)
                anim_pal = mapper

                for i in range(0, count):
                    self.img = self.animation_rds[self.variable].isel(time=i).to_masked_array()
                    self.img_cleaned = np.where(self.img < 0, np.nan, self.img)
                    self.apply_filter_rast()
                    self.calc_rast()

                    # img_stack_normalized = (self.img_cleaned - np.nanmin(self.img_cleaned)) / (
                    #     np.nanmax(self.img_cleaned) - np.nanmin(self.img_cleaned))

                    stack_lst.append(Image.fromarray(np.uint8(anim_pal.to_rgba(self.img_cleaned) * 255)))


                frames = [self.get_response_image(i) for i in stack_lst]


                self.geo_file_rast = frames
                self.data_type = "animation"
                self.raster_range = [minimum.item(), maximum.item()]
                self.raster_stats = {"length":len(frames)}

                self.animation_rds.close()
                self.rds.close()
                shutil.rmtree(self.xarr_path)

            else:
                with rasterio.open(self.path) as src:

                    self.check_crs_rast()

                    self.img = src.read(band)
                    self.raster_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]

                    self.clip_rast(src)

                    # <0 need to be changed
                    self.img_cleaned = np.where(self.img < 0, np.nan, self.img)

                    self.apply_filter_rast()

                    self.calc_rast()

                    self.extract_rast(src,band)

                if np.nanmin(self.img_cleaned)==0.0 and np.nanmax(self.img_cleaned)==0.0:
                    img_normalized = self.img_cleaned
                else:
                    img_normalized = (self.img_cleaned - np.nanmin(self.img_cleaned)) / (
                        np.nanmax(self.img_cleaned) - np.nanmin(self.img_cleaned))

                img_color = Image.fromarray(np.uint8(pal(img_normalized) * 255))


                self.raster_stats = {"mean": int(np.nanmean(self.img_cleaned)),
                                           "sum": int(np.nansum(self.img_cleaned))}
                self.geo_file_rast = self.get_response_image(img_color)
                self.raster_range = [np.nanmin(self.img_cleaned), np.nanmax(self.img_cleaned)]

            if xarr and not self.download:
                shutil.rmtree(self.dirpath)

    def set_up_xarray(self):

        self.xarr_path = tempfile.mkdtemp()
        self.rds = self.path
        self.path = f"{self.xarr_path}//temp_data.nc"

        self.rds.to_netcdf(self.path)
        self.rds = xarray.open_dataset(self.path)


        if self.time_cnt!="":
            rds_single = self.rds[self.variable].isel(time=int(self.time_cnt)).to_dataset()

            self.path = f"{self.dirpath}//dataset.tif"
            rds_single.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rio.to_raster(self.path)

            self.rds.close()
            shutil.rmtree(self.xarr_path)

        else:
            rds_single = self.rds[self.variable].isel(time=0).to_dataset()
            self.path = f"{self.dirpath}//dataset.tif"

            rds_single.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True).rio.to_raster(self.path)
            self.animation_rds = self.rds[self.variable].to_dataset()

        with rasterio.open(self.path) as src:

            self.check_crs_rast()
            self.raster_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]


        coord_names = list(self.rds.coords._names)
        animated_var = [i for i in coord_names if "lat" not in i and "lon" not in i]

        if "time" not in animated_var:
            self.time = "false"
        else:
            self.time = [f"{str(pd.Timestamp(i).year)}-{str(pd.Timestamp(i).month)}-{str(pd.Timestamp(i).day)} {str(pd.Timestamp(i).hour)}:00" for i in self.rds.coords[animated_var[0]].values]



    def pal_rast(self):
        if self.palette == "viridis":
            pal = plt.get_cmap("viridis",256)
        elif self.palette == "magma":
            pal = plt.get_cmap("magma",256)
        elif self.palette == "reds":
            pal = plt.get_cmap("Reds",256)
        else:
            pal = plt.get_cmap("Blues",256)

        return pal

    def extract_shp(self):
        if self.extract != "none":
            if self.extract == "single":
                if self.marker_loc != False:
                    lat = self.marker_loc["lat"]
                    lng = self.marker_loc["lng"]
                    pts = Point(lng, lat)
                    marker_gpd = gpd.GeoDataFrame(index=[0], geometry=[pts]).set_crs(epsg=4326)
                    joined_gpd = gpd.tools.sjoin(marker_gpd, self.file, how="inner")
                    if joined_gpd.shape[0] == 0:
                        self.error = "The marker does not intersect with the data set"
                    else:
                        val = joined_gpd[self.variable][0]
                        self.value = int(val)
                else:
                    self.error = "A marker must be drawn to receive single value"

            elif self.extract in ["sum", "mean"]:
                if self.boundaries != "false":
                    boundaries_shp = self.get_boundaries(self.boundaries)
                    self.file = gpd.clip(self.file, boundaries_shp)
                    joined_gpd = gpd.tools.sjoin(boundaries_shp, self.file, how="inner")
                    if joined_gpd.shape[0] == 0:
                        self.error = "The drawn shape does not intersect with the data set"
                    else:
                        if self.extract == "mean":
                            val = joined_gpd[self.variable].mean()
                        else:
                            val = joined_gpd[self.variable].sum()
                        self.value = int(val)
                else:
                    self.error = f"A polygon or rectangle must be drawn to receive the {self.extract}"
            self.data_type = "false"

    def extract_rast(self,src,band):
        if self.extract != "none":
            if self.extract == "single":
                if self.marker_loc != False:
                    lat = self.marker_loc["lat"]
                    lng = self.marker_loc["lng"]
                    val = [i[0] for i in src.sample([(lng, lat)])][0]
                    if val == -32768:
                        self.error = "The marker does not intersect with the data set"
                    else:
                        self.value = int(val)
                else:
                    self.error = "A marker must be drawn to receive single value"
            elif self.extract in ["sum", "mean"]:
                if self.boundaries != "false":
                    boundaries_rast = self.get_boundaries(self.boundaries)
                    bound_box = gpd.GeoDataFrame({"id": 1, "geometry": [box(*self.raster_bounds)]}).set_crs(
                        epsg=4326).intersects(boundaries_rast)
                    if not bound_box[0]:
                        self.error = "The polygon or rectangle does not intersect with the data set"
                    else:
                        if self.extract == "mean":
                            val = round(np.nanmean(np.where(src.read(band) < 0, np.nan, src.read(band))), 1)
                        else:
                            val = round(np.nansum(np.where(src.read(band) < 0, np.nan, src.read(band))), 1)
                        self.value = int(val)
                else:
                    self.error = f"A polygon or rectangle must be drawn to receive the {self.extract}"

            self.data_type = "false"

    def calc_shp(self):
        if self.calc_operator != "":
            try:
                self.file = self.file.assign(
                    temp_val=lambda df: df[self.variable].map(lambda x: eval(f"{x}{self.calc_operator}")))
                self.file.drop([f"{self.variable}"], axis=1, inplace=True)
                self.file.rename(columns={"temp_val": f"{self.variable}"}, inplace=True)
            except:
                self.error = "No valid calculation operator"

    def calc_rast(self):
        if self.calc_operator != "":
            try:
                self.img_cleaned = eval(f"self.img_cleaned{self.calc_operator}")
            except:
                self.error = "No valid calculation operator"

    def apply_filter_shp(self):
        filter = self.get_filter()
        if filter != f"(self.file.{self.variable})":
            try:
                self.file = eval(f"self.file[{filter}]")
            except:
                self.error = "No valid filter operator"

    def apply_filter_rast(self):
        filter = self.get_filter()
        if filter != "(self.img_cleaned)":
            try:
                self.img_cleaned = np.where(eval(f"{filter}"), self.img, np.nan)
            except:
                self.error = "No valid filter operator"


    def clip_shp(self):
        if self.boundaries != "false" and self.extract == "none":
            boundaries_shp = self.get_boundaries(self.boundaries)
            self.file = gpd.clip(self.file, boundaries_shp)
            if self.file.shape[0] == 0:
                self.error = "The polygon or rectangle shape does not intersect with the data set"

    def clip_rast(self,src):
        if self.boundaries != "false" and self.extract == "none":
            boundaries_rast = self.get_boundaries(self.boundaries)
            bound_box = gpd.GeoDataFrame({"id": 1, "geometry": [box(*self.raster_bounds)]}).set_crs(
                epsg=4326).intersects(
                boundaries_rast)
            if not bound_box[0]:
                self.error = "The polygon or rectangle does not intersect with the data set"
                self.data_type = "false"
            else:
                masked_img, affine = mask(src, boundaries_rast.geometry, invert=False)
                self.img = masked_img[0, :, :]

    def check_crs_shp(self):
        if self.file.crs == None:
            self.file = self.file.set_crs(epsg=4326)
        elif self.file.crs != "epsg:4326":
            self.file = self.file.to_crs(epsg=4326)

    def check_crs_rast(self):
        pass
        # if src.crs == None:
        #     src.crs = rasterio.crs.CRS({"init": "epsg:4326"})
        # elif src.crs != "EPSG:4326":
        #     pass


    def get_filter(self):
        filter_list = re.split("\&|\|", self.filter_operator_temp)
        filter_add = re.findall("\&|\|", self.filter_operator_temp)
        if self.data_type=="shapefile":
            filter_type = f"self.file.{self.variable}"
        else:
            filter_type = "self.img_cleaned"
        filter_operator = "".join(
                "(" + filter_type + filter_list[i] + ")" + filter_add[i] for i in range(0, len(filter_add))) \
                              + "(" + filter_type + filter_list[-1] + ")"
        return filter_operator

    def get_boundaries(self,boundaries):
        boundaries = gpd.read_file(boundaries)
        return boundaries

    def get_response_image(self,pil_img,format="PNG"):
        byte_arr = io.BytesIO()
        if format=="GIF":
            pil_img.save(byte_arr, format=format,save_all=True)
        else:
            pil_img.save(byte_arr, format=format)
        pil_data = byte_arr.getvalue()
        image_data = base64.b64encode(pil_data)
        if not isinstance(image_data, str):
            image_data = image_data.decode()
        return f"data:image/{format.lower()};base64," + image_data

    def get_json(self):

        xarr = self.type == "download"

        return {"geo_file_shp": self.geo_file_shp,
               "geo_file_rast": self.geo_file_rast,
               "error":self.error,
               "raster_bounds":self.raster_bounds,
               "raster_range":self.raster_range,
               "data_type":self.data_type,
               "value":self.value,
               "raster_stats":self.raster_stats,
               "palette":self.palette,
                "time":self.time,
                "xarr":xarr}


    def get_graphJSON(self):

        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y})

        if self.data_type == "shapefile":
            self.file = gpd.read_file(self.path)
            self.file.columns = self.file.columns.str.lower()
            self.file["prec"] = [random.randint(0, 10) for x in range(0, self.file.shape[0])]
            self.file["temp"] = [random.randint(15, 30) for x in range(0, self.file.shape[0])]

            data = [
                go.Histogram(x=self.file[self.variable], xbins=dict(size=0.5))
            ]
            graph = go.Figure(data)
            graph.update_layout(template="plotly_dark",
                                title="Dataset A",
                                yaxis_title="Counts",
                                xaxis_title="Values")

        else:

            with rasterio.open(self.path) as src:
                band = int(re.compile("(\d+)").search(self.variable).group(1))
                self.img = src.read(band)

            img_cleaned = np.where(self.img < 0, False, True)
            data = [
                go.Histogram(x=self.img[img_cleaned])
            ]
            graph = go.Figure(data)
            graph.update_layout(template="plotly_dark",
                                title="Dataset B",
                                yaxis_title="Counts",
                                xaxis_title="Values")

        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def get_xarr(self):
        if self.type=="download":

            xarr_path = tempfile.mkdtemp()

            with rasterio.open(self.path) as src:

                kwargs = src.meta

                self.apply_filter_rast()

                self.calc_rast()

                output_path = xarr_path+"\\xarr.tif"

                with rasterio.open(output_path,"w",**kwargs) as out:
                    out.write_band(1,self.img_cleaned)

        shutil.rmtree(self.dirpath)
        return output_path


def create_roi(boundaries):

    roi = gpd.read_file(boundaries).set_crs(epsg=4326)

    return roi.to_json()
