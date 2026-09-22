import xarray as xr
import pandas as pd
import h5py
import numpy as np
from abc import ABC, abstractmethod
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import datetime as dt
from geopy.distance import geodesic

class Atlid(ABC):
    """ 
    A class to represent the lidar flown on the EarthCARE satellite.
    Atlid is an Abstract Base Class - meaning we always require a more specific class 
    to be instantiated - ie you have to call Atlid_l2a(), you can't just call Atlid()

    Parameters
    ----------
    data : xarray.Dataset()
        Lidar data and attributes

    """
    @abstractmethod     # this stops you from being able to make a new generic lidar
    def __init__(self):
        """
        This is an abstract method since only inherited classes will be used to instantiate Lidar objects.
        """
        self.name = None
        self.data = None

    def despeckle(self, data_array=None, sigma=0.2):
        """
        Despeckle the lidar data by applying a Gaussion filter with a given standard deviation

        Parameters
        ----------
        data_array : xarray.DataArray()
            The field to despeckle
        sigma : float
            The standard deviation to apply to the gaussian filter
        """
        temp_datacopy =  data_array.copy()

        # run the data array through a gaussian filter
        temp_datafiltered = gaussian_filter(temp_datacopy, sigma)

        # np.isfinite returns values that are not NAN or INF
        return data_array.where(np.isfinite(temp_datafiltered))

class Atlid_l2a(Atlid):
    def __init__(self, filepath):
        if 'EBD_2A' in filepath:
            self.name = 'ATLID L2A A-EBD'
            self.data = self.readfile_aed(filepath)

    def readfile_aed(self, filepath):
        """
        xarray.Dataset of lidar variables and attributes
    
        Dimensions:
            - gate: xarray.DataArray(float) - The lidar gate/sample number  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp
    
        Coordinates:
            - gate (gate): xarray.DataArray(float) - The lidar gate/sample number  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
    
        Variables:
            - rayleigh_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated molecular backscatter profile (km**-1 sr**-1)
            - mie_total_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated particulate backscatter profile (km**-1 sr**-1)
            - total_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated total (molecular + particulate) backscatter profile (km**-1 sr**-1)
            - particle_linear_depol_ratio_355nm (gate, time) : xarray.DataArray(float) - Particulate linear depolarization ratio profile (#)
        """
        hdf = h5py.File(filepath, 'r')
        
        # coordinates
        dt = pd.to_datetime(
            hdf['ScienceData']['time'][:], unit='s',
            origin=pd.Timestamp('2000-01-01 00:00:00')
        ).to_numpy()
        #print (hdf['ScienceData'].keys())
        lat = xr.DataArray(
            data = hdf['ScienceData']['latitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Latitude',
                units = 'degrees North'
            )
        )
        lon = xr.DataArray(
            data = hdf['ScienceData']['longitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Longitude',
                units = 'degrees East'
            )
        )
        
        elv = xr.DataArray(
            data = hdf['ScienceData']['elevation'][:],
            dims = 'time',
            attrs = dict(
                description = 'Elevation',
                units = 'm'
            )
        )
        
        num_layers = xr.DataArray(
            data = hdf['ScienceData']['number_of_layers'][:],
            dims = 'time',
            attrs = dict(
                description = 'Number of layers',
                units = 'unitless'
            )
        )
    
        # 2D variables
        hght = xr.DataArray(
            data = hdf['ScienceData']['height'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Altitude above ground level',
                units = 'm'
            )
        )

        mask = xr.DataArray(
            data = hdf['ScienceData']['simple_classification'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Simple mask: -3 missing data, -2 surface, -1 both mie and ray attenuated, 0 clear, 1 ice, 2 water, 3 aerosol, 4 St cloud, St aerosol',
                units = 'unitless'
            )
        )

        layer_bot = xr.DataArray(
            data = hdf['ScienceData']['layer_bottom_index'][:].T,
            dims = ('layers', 'time'),
            attrs = dict(
                description = 'Index of Layer bottoms',
                units = 'unitless'
            )
        )
        
        layer_top = xr.DataArray(
            data = hdf['ScienceData']['layer_top_index'][:].T,
            dims = ('layers', 'time'),
            attrs = dict(
                description = 'Index of Layer tops',
                units = 'unitless'
            )
        )
    
        att_bsc_ray = xr.DataArray(
            data = 1000. * hdf['ScienceData']['rayleigh_attenuated_backscatter_355nm'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Rayleigh (molecular) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )
        att_bsc_ray = att_bsc_ray.where((att_bsc_ray >= 0.) & (att_bsc_ray < 10.))
        
        att_bsc_mie = xr.DataArray(
            data = 1000. * hdf['ScienceData']['mie_total_attenuated_backscatter_355nm'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Mie (particulate) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )
        att_bsc_mie = att_bsc_mie.where((att_bsc_mie >= 0.) & (att_bsc_mie < 10.))
    
        atb = xr.DataArray(
            data = att_bsc_ray.values + att_bsc_mie.values,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Total (particulate + molecular) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )

        atb_total = xr.DataArray(
            data = 1000. * hdf['ScienceData']['total_attenuated_backscatter_355nm'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Mie (particulate) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )

        hdf.close()
        data_vars = dict(
            height = hght,
            num_lay = num_layers,
            mask = mask,
            elevation = elv,
            layer_bot = layer_bot, 
            layer_top = layer_top,
            att_bsc_ray = att_bsc_ray,
            att_bsc_mie = att_bsc_mie,
            att_bsc_total = atb, 
            att_total = atb_total
        )
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'gate': np.arange(hght.values.shape[0]),
                'time': dt, 'lat': lat, 'lon': lon,
                'layers': np.arange(layer_bot.values.shape[0])
            }
        )
        return ds

class Atlid_l2a_ATC(Atlid):
    def __init__(self, filepath):
        if 'TC__2A' in filepath:
            self.name = 'ATLID L2A A-ATC'
            self.data = self.readfile_aed(filepath)

    def readfile_aed(self, filepath):
        """
        xarray.Dataset of lidar variables and attributes
    
        Dimensions:
            - gate: xarray.DataArray(float) - The lidar gate/sample number  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp
    
        Coordinates:
            - gate (gate): xarray.DataArray(float) - The lidar gate/sample number  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
    
        Variables:
            - rayleigh_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated molecular backscatter profile (km**-1 sr**-1)
            - mie_total_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated particulate backscatter profile (km**-1 sr**-1)
            - total_attenuated_backscatter_355nm (gate, time) : xarray.DataArray(float) - Attenuated total (molecular + particulate) backscatter profile (km**-1 sr**-1)
            - particle_linear_depol_ratio_355nm (gate, time) : xarray.DataArray(float) - Particulate linear depolarization ratio profile (#)
        """
        hdf = h5py.File(filepath, 'r')
        
        # coordinates
        dt = pd.to_datetime(
            hdf['ScienceData']['time'][:], unit='s',
            origin=pd.Timestamp('2000-01-01 00:00:00')
        ).to_numpy()
        #print (hdf['ScienceData'].keys())
        lat = xr.DataArray(
            data = hdf['ScienceData']['latitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Latitude',
                units = 'degrees North'
            )
        )
        lon = xr.DataArray(
            data = hdf['ScienceData']['longitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Longitude',
                units = 'degrees East'
            )
        )
        
        elv = xr.DataArray(
            data = hdf['ScienceData']['elevation'][:],
            dims = 'time',
            attrs = dict(
                description = 'Elevation',
                units = 'm'
            )
        )
        
        # 2D variables
        hght = xr.DataArray(
            data = hdf['ScienceData']['height'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Altitude above ground level',
                units = 'm'
            )
        )

        mask = xr.DataArray(
            #data = hdf['ScienceData']['simple_classification'][:].T,
            data = hdf['ScienceData']['classification'][:].T,
            #data = hdf['ScienceData']['classification_medium_resolution'][:].T,
            #data = hdf['ScienceData']['classification_low_resolution'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Simple mask: -3 missing data, -2 surface, -1 both mie and ray attenuated, 0 clear, 1 ice, 2 water, 3 aerosol, 4 St cloud, St aerosol',
                units = 'unitless'
            )
        )

        qc = xr.DataArray(
            data = hdf['ScienceData']['quality_status'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = '0: Good, 1: likely good, 2: likely bad, 3: bad, 4: missing',
                units = 'unitless'
            )
        )

        hdf.close()
        data_vars = dict(
            height = hght,
            mask = mask,
            elevation = elv,
            qc = qc
        )
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'gate': np.arange(hght.values.shape[0]),
                'time': dt, 'lat': lat, 'lon': lon,
                #'layers': np.arange(layer_bot.values.shape[0])
            }
        )
        return ds

class Atlid_l1(Atlid):
    def __init__(self, filepath):
        if 'NOM_1B' in filepath:
            self.name = 'ATLID L1B NOM'
            self.data = self.readfile_nom(filepath)

    def readfile_nom(self, filepath):
        """
        xarray.Dataset of lidar variables and attributes
    
        Dimensions:
            - gate: xarray.DataArray(float) - The lidar gate/sample number  
            - time: np.array(np.datetime64[ns]) - The UTC time stamp
    
        Coordinates:
            - gate (gate): xarray.DataArray(float) - The lidar gate/sample number  
            - height (range, time): xarray.DataArray(float) - Altitude of each range gate (m)  
            - time (time): np.array(np.datetime64[ns]) - The UTC time stamp  
            - lat (time): xarray.DataArray(float) - Latitude (degrees)
            - lon (time): xarray.DataArray(float) - Longitude (degrees)
    
        Variables:
            - rayleigh_attenuated_backscatter (gate, time) : xarray.DataArray(float) - Attenuated molecular backscatter profile (km**-1 sr**-1)
            - mie_attenuated_backscatter (gate, time) : xarray.DataArray(float) - Attenuated particulate backscatter profile (km**-1 sr**-1)
            - total_attenuated_backscatter (gate, time) : xarray.DataArray(float) - Attenuated total (molecular + particulate) backscatter profile (km**-1 sr**-1)
        """
        hdf = h5py.File(filepath, 'r')
        
        # coordinates
        dt = pd.to_datetime(
            hdf['ScienceData']['time'][:], unit='s',
            origin=pd.Timestamp('2000-01-01 00:00:00')
        ).to_numpy()
        lat = xr.DataArray(
            data = hdf['ScienceData']['sensor_latitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Satellite latitude',
                units = 'degrees North'
            )
        )
        lon = xr.DataArray(
            data = hdf['ScienceData']['sensor_longitude'][:],
            dims = 'time',
            attrs = dict(
                description = 'Satellite longitude',
                units = 'degrees East'
            )
        )
    
        # 2D variables
        slat = xr.DataArray(
            data = hdf['ScienceData']['sample_latitude'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Sample (pixel) latitude',
                units = 'degrees North'
            )
        )
        slon = xr.DataArray(
            data = hdf['ScienceData']['sample_longitude'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Sample (pixel) longitude',
                units = 'degrees East'
            )
        )
        hght = xr.DataArray(
            data = hdf['ScienceData']['sample_altitude'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Sample (pixel) altitude above ground level',
                units = 'm'
            )
        )
    
        att_bsc_ray = xr.DataArray(
            data = 1000. * hdf['ScienceData']['rayleigh_attenuated_backscatter'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Rayleigh (molecular) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )
        att_bsc_ray = att_bsc_ray.where((att_bsc_ray >= 0.) & (att_bsc_ray < 10.))
        
        att_bsc_mie = xr.DataArray(
            data = 1000. * hdf['ScienceData']['mie_attenuated_backscatter'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Mie (particulate) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )
        att_bsc_mie = att_bsc_mie.where((att_bsc_mie >= 0.) & (att_bsc_mie < 10.))

        atb_total = xr.DataArray(
            data = 1000. * hdf['ScienceData']['total_attenuated_backscatter_355nm'][:].T,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Mie (particulate) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )
        
        atb = xr.DataArray(
            data = att_bsc_ray.values + att_bsc_mie.values,
            dims = ('gate', 'time'),
            attrs = dict(
                description = 'Total (particulate + molecular) attenuated backscatter',
                units = 'km**-1 sr**-1'
            )
        )

        hdf.close()
        data_vars = dict(
            sample_lat = slat,
            sample_lon = slon,
            height = hght,
            att_bsc_ray = att_bsc_ray,
            att_bsc_mie = att_bsc_mie,
            att_bsc_total = atb_total
            #att_total = atb_total
        )
        ds = xr.Dataset(
            data_vars = data_vars,
            coords = {
                'gate': np.arange(hght.values.shape[0]),
                'time': dt, 'lat': lat, 'lon': lon
            }
        )
        return ds

