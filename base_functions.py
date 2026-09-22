import xarray as xr
import pandas as pd
import h5py
import numpy as np
from abc import ABC, abstractmethod
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import datetime as dt
from geopy.distance import geodesic
from atlid_functions_mask import *

def closest_approach(satellite_coordinates, airplane_coordinates, sat_time, air_time):

    # Satellite and Airplane coordinates are in degrees and must be a tuple with the format (Lat, Lon)
    # I want to know the moment in which EarthCare was the closest to the airplane

    from geopy.distance import geodesic

    # Satellite coordinates
    sat_lons = satellite_coordinates[1]
    sat_lats = satellite_coordinates[0]

    # Airplane coordinates
    air_lons = airplane_coordinates[1]
    air_lats = airplane_coordinates[0]

    # Step 1: Interpolate airplane position to satellite times
    from scipy.interpolate import interp1d

    # Convert times to seconds for interpolation
    air_time_sec = (air_time - air_time[0]) / np.timedelta64(1, 's') 
    sat_time_sec = (sat_time - air_time[0]) / np.timedelta64(1, 's')

    # Interpolators
    interp_lon = interp1d(air_time_sec, air_lons, bounds_error=False, fill_value='extrapolate')
    interp_lat = interp1d(air_time_sec, air_lats, bounds_error=False, fill_value='extrapolate')

    # Airplane positions at satellite times
    air_lons_interp = interp_lon(sat_time_sec)
    air_lats_interp = interp_lat(sat_time_sec)

    # Step 2: Compute distances at each time step
    distances = np.array([
        geodesic((sat_lat, sat_lon), (air_lat, air_lon)).meters
        for sat_lat, sat_lon, air_lat, air_lon in zip(sat_lats, sat_lons, air_lats_interp, air_lons_interp)
    ])

    # Step 3: Find closest approach
    min_idx = np.argmin(distances)
    closest_distance = distances[min_idx]
    closest_time = sat_time[min_idx]
    closest_sat_pos = (sat_lats[min_idx], sat_lons[min_idx])
    closest_air_pos = (air_lats_interp[min_idx], air_lons_interp[min_idx])

    print(f"Closest distance: {closest_distance:.2f} meters")
    print(f"Time of closest approach: {closest_time}")
    print(f"Satellite position: {closest_sat_pos}")
    print(f"Airplane position: {closest_air_pos}")

    # Now: Find the closest index in the airplane array to that same timestamp
    closest_air_idx = np.argmin(np.abs(air_time - sat_time[min_idx]))

    print(f"Satellite index at closest approach: {min_idx}")
    print(f"Airplane index at closest approach: {closest_air_idx}")
    
    # returning the index of the closest approach for the satellite and airplane respectively
    return min_idx, closest_air_idx

def average_lidar_data(a,ne,nd,trunc=0):
    """Average lidar data (or other arrays).
       INPUTS: 
               a  -> the input np array. MUST BE 2 DIMENSIONS!
               ne -> the number of elements to average
               nd -> the dimension # to be averaged (start @0)   
               trunc -> -1: Don't truncate. Any leftovers averaged into last element.
                         0: Truncate any amount of leftover. Default.
                         >=1: Number represents the min req to NOT truncate.
       
       LAST MODIFIED: 2/27/23                           
    """
    import_fail = False
    try:
        from skimage.measure import block_reduce
    except:
        print('Import of skimage.measure.block_reduce failed.')
        print('Using pre-Feb2023 method of averaging...')
        import_fail = True
    
    orig_shape = a.shape
    ne_orig = ne
    if nd == 0:
        ne = (ne,1)
    else:
        ne = (1,ne)
    
    a = block_reduce(a, block_size=ne, func=np.nanmean) 
    leftover = orig_shape[nd] % ne_orig
    gotta_trunc = False
    if ((trunc == 0) and (leftover != 0)): gotta_trunc = True
    if ((trunc >= 1) and (leftover < trunc)): gotta_trunc = True
    if gotta_trunc:
        if nd == 0: 
            a = a[:-1,:]
        else:
            a = a[:,:-1]
    
    return a

from scipy.spatial import cKDTree

# Convert lat/lon to 3D Cartesian coordinates (for KDTree)
def latlon_to_xyz(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    R = 6371  # Earth's radius in kilometers
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.stack((x, y, z), axis=-1)

import numpy as np
from scipy.spatial import cKDTree

def find_similar_path(satellite_coordinates, airplane_coordinates, threshold_km=10, min_segment_length=3):
    """
    Finds the longest continuous similar path between satellite and airplane trajectories.
    Ignores single-point overlaps and short segments below `min_segment_length`.
    """

    sat_lats, sat_lons = satellite_coordinates
    air_lats, air_lons = airplane_coordinates

    air_xyz = latlon_to_xyz(air_lats, air_lons)
    sat_xyz = latlon_to_xyz(sat_lats, sat_lons)

    tree = cKDTree(air_xyz)
    distances, indices = tree.query(sat_xyz, distance_upper_bound=threshold_km)

    # Create list of (sat_idx, air_idx) pairs that are within the threshold
    close_pairs = [(i, idx) for i, (d, idx) in enumerate(zip(distances, indices)) if np.isfinite(d)]

    if not close_pairs:
        print("No close positions found within the distance threshold.")
        return None

    # Group into consecutive segments based on satellite index
    segments = []
    current_segment = [close_pairs[0]]

    for i in range(1, len(close_pairs)):
        prev_sat_idx, _ = close_pairs[i - 1]
        curr_sat_idx, _ = close_pairs[i]

        if curr_sat_idx == prev_sat_idx + 1:
            current_segment.append(close_pairs[i])
        else:
            if len(current_segment) >= min_segment_length:
                segments.append(current_segment)
            current_segment = [close_pairs[i]]

    # Append the last segment if it's long enough
    if len(current_segment) >= min_segment_length:
        segments.append(current_segment)

    if not segments:
        print("No valid overlapping segments found.")
        return None

    # Select the longest segment
    longest_segment = max(segments, key=len)

    first_sat_idx, first_air_idx = longest_segment[0]
    last_sat_idx, last_air_idx = longest_segment[-1]

    return (first_sat_idx, last_sat_idx), (first_air_idx, last_air_idx)

def Map(satellite_coordinates, airplane_coordinates, sat_time = None, air_time = None, 
        sn = 'EarthCare', an = 'CPL', close_approach = False):
    
    # sn stands for Satellite Name, default is EarthCare
    # an stands for Airplane Name, default is CPL
    # Satellite and Airplane coordinates are in degrees and must be a tuple with the format (Lat, Lon)
    Lat_sat = satellite_coordinates[0]
    Lon_sat = satellite_coordinates[1]

    Lat_air = airplane_coordinates[0]
    Lon_air = airplane_coordinates[1] 
    
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    ax.plot(Lon_sat, Lat_sat, color='red', linewidth=4, transform=ccrs.PlateCarree(), label = sn)
    ax.plot(Lon_air, Lat_air, color='m', linewidth=4, transform=ccrs.PlateCarree(), label = an)

    # Set extent: [west, east, south, north]
    ax.set_extent([-130, -95, 24, 50], crs=ccrs.PlateCarree())

    # Add features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True)
    if close_approach == True:
        # Retrieving the indices for the closest approach
        min_idx, closest_air_idx = closest_approach(satellite_coordinates, airplane_coordinates,
                                                   sat_time, air_time)
        plt.scatter(Lon_air[closest_air_idx], Lat_air[closest_air_idx], 
                    marker = '*', color = 'yellow', ec = 'k', zorder = 4,
                    s = 250, label = 'Closest Approach')
        
        sat_indices, air_indices = find_similar_path((Lat_sat, Lon_sat), (Lat_air, Lon_air), threshold_km = 10)
        
        ax.plot(Lon_sat[sat_indices[0]: sat_indices[1]], Lat_sat[sat_indices[0]: sat_indices[1]], color='blue', linewidth=4, 
                transform=ccrs.PlateCarree(), label = 'Coincident track')
        
    plt.legend(fontsize = 14)
    plt.show()
    
    if close_approach == True:
        return  min_idx, closest_air_idx

def adjust_time(time_list, time_minutes, time_seconds, time_hours = 0):
    adjust_time_list = (time_list + 
                        np.timedelta64(time_hours, 'h') + 
                        np.timedelta64(time_minutes, 'm') + 
                        np.timedelta64(time_seconds, 's'))
    return adjust_time_list

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LogNorm, ListedColormap, Normalize
import matplotlib.colors as mcolors

def make_cmap(mycolors, position=None, bit=False):
    """
    Custom Colormaps for Matplotlib

    This program shows how to implement make_cmap which is a function that
    generates a colorbar.  If you want to look at different color schemes,
    check out https://kuler.adobe.com/create.

    PROGRAMMER(S)
      Chris Slocum

    REVISION HISTORY
      20130411 -- Initial version created
      20140313 -- Small changes made and code posted online
      20140320 -- Added the ability to set the position of each color

      make_cmap takes a list of tuples which contain RGB values. The RGB
      values may either be in 8-bit [0 to 255] (in which bit must be set to
      True when called) or arithmetic [0 to 1] (default). make_cmap returns
      a cmap with equally spaced colors.
      Arrange your tuples so that the first color is the lowest value for the
      colorbar and the last is the highest.
      position contains values from 0 to 1 to dictate the location of each color.
    """

    bit_rgb = np.linspace(0,1,256)
    if position is None:
      position = np.linspace(0,1,len(mycolors))

    if bit:
      for i in range(len(mycolors)):
        mycolors[i] = (bit_rgb[mycolors[i][0]], bit_rgb[mycolors[i][1]],
                       bit_rgb[mycolors[i][2]])

    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, mycolors):
      cdict['red'].append((pos, color[0], color[0]))
      cdict['green'].append((pos, color[1], color[1]))
      cdict['blue'].append((pos, color[2], color[2]))

    cmap = mcolors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def get_stevepalm_cmap():
    """ Returns a custom colormap developed by Steve Palm """

    # Define the RGB values
    sp_colors = [
      (  0,   0,  16), (  0,   4,  20), (  0,   8,  24), (  0,  12,  28),
      (  0,  16,  32), (  4,  16,  40), (  8,  16,  48), ( 12,  16,  56),
      ( 16,  16,  64), ( 16,  20,  68), ( 16,  24,  72), ( 16,  28,  76),
      ( 16,  32,  80), ( 16,  36,  84), ( 16,  40,  88), ( 16,  44,  92),
      ( 16,  48, 100), ( 16,  52, 104), ( 16,  56, 108), ( 16,  60, 112),
      ( 16,  64, 116), ( 16,  64, 120), ( 16,  64, 124), ( 16,  64, 128),
      ( 16,  64, 132), ( 20,  68, 136), ( 24,  72, 140), ( 28,  76, 144),
      ( 32,  80, 148), ( 32,  84, 152), ( 32,  88, 156), ( 32,  92, 160),
      ( 32, 100, 164), ( 32, 100, 168), ( 32, 100, 172), ( 32, 100, 176),
      ( 32, 100, 184), ( 32, 104, 188), ( 32, 108, 192), ( 32, 112, 196),
      ( 32, 116, 200), ( 36, 120, 204), ( 40, 124, 208), ( 44, 128, 212),
      ( 48, 132, 216), ( 48, 136, 224), ( 48, 140, 232), ( 48, 144, 240),
      ( 48, 148, 248), ( 48, 148, 244), ( 48, 148, 240), ( 48, 148, 236),
      ( 48, 148, 232), ( 48, 152, 228), ( 48, 156, 224), ( 48, 160, 220),
      ( 48, 164, 216), ( 52, 168, 212), ( 56, 172, 208), ( 60, 176, 204),
      ( 64, 184, 200), ( 64, 188, 196), ( 64, 192, 192), ( 64, 196, 188),
      ( 64, 200, 184), ( 64, 200, 176), ( 64, 200, 172), ( 64, 200, 168),
      ( 64, 200, 164), ( 64, 204, 160), ( 64, 208, 156), ( 64, 212, 152),
      ( 64, 216, 148), ( 68, 216, 140), ( 72, 216, 132), ( 76, 216, 124),
      ( 80, 216, 116), ( 80, 220, 112), ( 80, 224, 108), ( 80, 228, 104),
      ( 80, 232, 100), ( 80, 236,  92), ( 80, 240,  88), ( 80, 244,  84),
      ( 80, 248,  80), ( 80, 244,  76), ( 80, 240,  72), ( 80, 236,  68),
      ( 80, 232,  64), ( 80, 232,  60), ( 80, 232,  56), ( 80, 232,  52),
      ( 80, 232,  48), ( 84, 228,  44), ( 88, 224,  40), ( 92, 220,  36),
      (100, 216,  32), (100, 216,  24), (100, 216,  16), (100, 216,   8),
      (100, 216,   0), (100, 212,   4), (100, 208,   8), (100, 204,  12),
      (100, 200,  16), (100, 196,  20), (100, 192,  24), (100, 188,  28),
      (100, 184,  32), (104, 176,  40), (108, 172,  48), (112, 168,  56),
      (116, 164,  64), (116, 160,  68), (116, 156,  72), (116, 152,  76),
      (116, 148,  80), (116, 148,  84), (116, 148,  88), (116, 148,  92),
      (116, 148, 100), (116, 144, 104), (116, 140, 108), (116, 136, 112),
      (116, 132, 116), (120, 128, 120), (124, 124, 124), (128, 120, 128),
      (132, 116, 132), (132, 112, 136), (132, 108, 140), (132, 104, 144),
      (132, 100, 148), (132, 100, 156), (132, 100, 164), (132, 100, 172),
      (132, 100, 184), (132,  92, 188), (132,  88, 192), (132,  84, 196),
      (132,  80, 200), (136,  76, 204), (140,  72, 208), (144,  68, 212),
      (148,  64, 216), (148,  64, 220), (148,  64, 224), (148,  64, 228),
      (148,  64, 232), (148,  60, 236), (148,  56, 240), (148,  52, 244),
      (148,  48, 248), (148,  44, 244), (148,  40, 240), (148,  36, 236),
      (148,  32, 232), (152,  32, 224), (156,  32, 216), (160,  32, 208),
      (164,  32, 200), (164,  28, 196), (164,  24, 192), (164,  20, 188),
      (164,  16, 184), (164,  12, 176), (164,   8, 172), (164,   4, 168),
      (164,   0, 164), (164,   4, 160), (164,   8, 156), (164,  12, 152),
      (164,  16, 148), (168,  16, 144), (172,  16, 140), (176,  16, 136),
      (184,  16, 132), (184,  20, 124), (184,  24, 116), (184,  28, 108),
      (184,  32, 100), (184,  36,  92), (184,  40,  88), (184,  44,  84),
      (184,  48,  80), (184,  52,  76), (184,  56,  72), (184,  60,  68),
      (184,  64,  64), (184,  64,  60), (184,  64,  56), (184,  64,  52),
      (184,  64,  48), (188,  68,  44), (192,  72,  40), (196,  76,  36),
      (200,  80,  32), (200,  84,  24), (200,  88,  16), (200,  92,   8),
      (200, 100,   0), (200, 104,   4), (200, 108,   8), (200, 112,  12),
      (200, 116,  16), (200, 116,  20), (200, 116,  24), (200, 116,  28),
      (200, 116,  32), (204, 120,  36), (208, 124,  40), (212, 128,  44),
      (216, 132,  48), (216, 136,  52), (216, 140,  56), (216, 144,  60),
      (216, 148,  64), (216, 148,  72), (216, 148,  80), (216, 148,  88),
      (216, 148, 100), (216, 152, 104), (216, 156, 108), (216, 160, 112),
      (216, 164, 116), (220, 168, 120), (224, 172, 124), (228, 176, 128),
      (232, 184, 132), (232, 188, 136), (232, 192, 140), (232, 196, 144),
      (232, 200, 148), (232, 200, 152), (232, 200, 156), (232, 200, 160),
      (232, 200, 164), (232, 204, 172), (232, 208, 180), (232, 212, 188),
      (232, 216, 200), (236, 220, 204), (240, 224, 208), (244, 228, 212),
      (248, 232, 216), (248, 236, 220), (248, 240, 224), (248, 244, 228),
      (248, 248, 232), (248, 248, 236), (248, 248, 240), (248, 248, 244),
      (248, 248, 248), (248, 248, 248), (248, 248, 248), (248, 248, 248)]

    # Make the color map
    return make_cmap(sp_colors, bit=True)

def extract_and_organize_data(filepath_ec, filepath_ec_atc, fileday_cpl, fileday_cpl_fe, range_height = True, layer1 = 0, layer2 = 12,
                             minutos = 15, minute1 = None, minute2 = None):
        # Extracting EarthCARE Level 1 and Level 2 data
    
    instance  = Atlid_l2a(filepath_ec)
    ATB355_EC = instance.data.att_total
    Alt_EC    = instance.data.height/1000.
    Elevation = instance.data.elevation/1000.
    #mask_feature = instance.data.mask
    num_layer = instance.data.num_lay
    lay_bot = instance.data.layer_bot
    lay_top = instance.data.layer_top

    instance2  = Atlid_l2a_ATC(filepath_ec_atc)
    Lat_EC    = instance2.data.lat.values
    Lon_EC    = instance2.data.lon.values
    time_EC   = instance2.data.time.values
    Alt_EC    = instance2.data.height/1000.
    Alt_EC_cp = instance2.data.height/1000.
    Elevation = instance2.data.elevation/1000.
    mask_feature = instance2.data.mask
    mask_feature0 = mask_feature.copy(deep = True)
    qc        = instance2.data.qc
    #mask_feature = mask_feature.where(qc <= 1,  np.nan)


    # Extracting CPL Level 1 and Level 2 data
    ds = Dataset(fileday_cpl)

    # This is the CPL Level 1 data
    ATB355_cpl = ds['ATB_355'][:] # Attenuated bz at 355 nm
    ATB532_cpl = ds['ATB_532'][:]  # Attenuated bz at 532 nm
    Mole355_cpl= ds['Mole_Back'][:][0,:,:] #phony_dim_0 = 355nm, phony_dim_1 = 532 nm, phony_dim_3 = 1064 nm 
    Mole532_cpl= ds['Mole_Back'][:][1,:,:] #phony_dim_0 = 355nm, phony_dim_1 = 532 nm, phony_dim_3 = 1064 nm 
    Lat_cpl    = ds['Latitude'][:]
    Lon_cpl    = ds['Longitude'][:]
    Alt_cpl    = ds['Bin_Alt'][:]
    Alt_cpl_cp = ds['Bin_Alt'][:]
    Hour_cpl   = ds['Hour'][:]
    Minute_cpl = ds['Minute'][:]
    Second_cpl = ds['Second'][:]
    Dem_cpl    = ds['DEM_laserspot'][:]/1000.
    day        = fileday_cpl[-20:-18]
    ds.close()

    # Creating a dataframe with the datetime of the CPL Flight
    df = pd.DataFrame({'year': np.transpose([2025]*len(Hour_cpl)),
                       'month': np.transpose([2]*len(Hour_cpl)),
                       'day': np.transpose([day]*len(Hour_cpl)),
                        'hour': Hour_cpl, 
                        'minutes': Minute_cpl, 
                        'seconds': Second_cpl})
    # Changing the dataframe format to be datetime format
    time_cpl = pd.to_datetime(df).values

    # this is the CPL Level 2 data
    ds = Dataset(fileday_cpl_fe)
    featype_cpl = ds['profile']['Feature_Type'][:]
    cphase_cpl   = ds['profile']['Cloud_Phase'][:]
    lat_fea        = ds['geolocation']['CPL_Latitude'][:,0]
    lon_fea        = ds['geolocation']['CPL_Longitude'][:,0]

    positions_cloud = np.where(featype_cpl == 1)
    positions_aerosol = np.where(featype_cpl == 3)
    featype_cpl[positions_cloud] = cphase_cpl[positions_cloud]
    featype_cpl[positions_aerosol] = 5.
    
    ds.close()

    # Computing the closest approach between EarthCARE and CPL and extracting index where they are closest
    min_idx, closest_air_idx = closest_approach((Lat_EC, Lon_EC), (Lat_cpl, Lon_cpl), sat_time = time_EC, air_time = time_cpl)

    # Let's take 5 minutes around the coincident time and a similar path for EarthCare
    start_time = time_cpl[closest_air_idx] - pd.Timedelta(minutes = minutos)
    end_time   = time_cpl[closest_air_idx] + pd.Timedelta(minutes = minutos)

    # These are the CPL indices
    indices = np.where((time_cpl >= start_time) & (time_cpl <= end_time))[0]

    # These are the EarthCARE indices
    indices_EC = np.where((Lat_EC >= np.min(Lat_cpl[indices])) & (Lat_EC <= np.max(Lat_cpl[indices])))[0]
    mean_height = np.mean(Dem_cpl[indices].data)

    # Now Let's average CPL Data to 1 km resolution, assuming a 200 m horizontal resolution
    # ======================================================================================
    # Selecting the number of profiles of CPL I need to average in order to get the desired horizontal resolution
    nhori = int(np.floor(1./0.2)) # 1 km / 0.2 km

    # Now Let's average CPL Data to 5 km resolution, assuming a 200 m horizontal resolution
    ATB355_cpl2 = average_lidar_data(ATB355_cpl, nhori, 0)
    ATB532_cpl2 = average_lidar_data(ATB532_cpl, nhori, 0)

    # Averaging the surface elevation
    Elv_cpl2 = np.array([Dem_cpl, Dem_cpl])
    Elv_cpl2 = average_lidar_data(Elv_cpl2.T, nhori, 0)
    
    # Now let's average the time dataset of CPL
    # Convert datetime64 to int for mean computation
    time_as_int = time_cpl.astype('int64')  # nanoseconds since epoch
    
    # Make it 2D to use with your function
    time_as_2d = time_as_int.reshape(1, -1)  # shape (1, N)
    
    # Apply your function to average every 5 values along dimension 1
    mean_time_int = average_lidar_data(time_as_2d, ne=nhori, nd=1)
    
    # Convert back to datetime64
    time_cpl2 = mean_time_int.astype('datetime64[ns]')

    # Lat CPL2
    Lat_cpl2 = np.array([Lat_cpl, Lat_cpl])
    Lat_cpl2 = average_lidar_data(Lat_cpl2.T, nhori, 0)

    # Computing coincident times between CPL and EarthCARE at 1 km resolution
    if day in ['12']:
        min_idx, closest_air_idx = closest_approach((Lat_EC, Lon_EC), (lat_fea[1:-1], lon_fea[1:-1]), sat_time = time_EC, 
            air_time = time_cpl2[0])
    elif day == '04':
        min_idx, closest_air_idx = closest_approach((Lat_EC, Lon_EC), (lat_fea, lon_fea), sat_time = time_EC, 
            air_time = time_cpl2[0])
    elif day in ['07', '10', '19', '20']:
    #else:
                min_idx, closest_air_idx = closest_approach((Lat_EC, Lon_EC), (lat_fea[:-1], lon_fea[:-1]), sat_time = time_EC, 
            air_time = time_cpl2[0])


    # Let's take 15 minutes around the coincident time and a similar path for EarthCare
    if minute1 == None:
        start_time2 = time_cpl2[0][closest_air_idx] - pd.Timedelta(minutes = minutos)
        end_time2   = time_cpl2[0][closest_air_idx] + pd.Timedelta(minutes = minutos)

        # Indices 2 corresponds to the coincident in 10 minutes period track for 1 km resolution 
        indices2 = np.where((time_cpl2[0] >= start_time2) & (time_cpl2[0] <= end_time2))[0]
    else:
        start_time2 = time_cpl2[0][closest_air_idx] + pd.Timedelta(minutes = minute1)
        end_time2   = time_cpl2[0][closest_air_idx] + pd.Timedelta(minutes = minute2)
        if (start_time2 > end_time2):
            indices2 = np.where((time_cpl2[0] >= end_time2) & (time_cpl2[0] <= start_time2))[0]
        else:
            indices2 = np.where((time_cpl2[0] >= start_time2) & (time_cpl2[0] <= end_time2))[0]


    # Computing the difference in time for the CPL track, from the closest approach
    diff_minutes = (time_cpl2[0] - time_cpl2[0][closest_air_idx]) / np.timedelta64(1, 'm')
    
    indices_EC2 = np.where((Lat_EC >= np.min(lat_fea[indices2])) & 
                           (Lat_EC <= np.max(lat_fea[indices2])))[0]
    mean_height2 = np.mean(Dem_cpl[indices2].data)
    
    return (Lat_cpl2, Lat_EC, ATB532_cpl2, indices2, Alt_cpl, Elv_cpl2, 
            closest_air_idx, Alt_EC, Lat_EC, indices_EC2, min_idx, ATB355_EC, Elevation,
            lat_fea, featype_cpl, mask_feature)