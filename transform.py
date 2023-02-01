#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ctypes
import datetime
import glob
import dill as pickle
import os
import platform

PHI0 = 0    # Angles phi are wrt to sattellite flying direction. Satellite flies
            # almost north, but 33 degrees west. Subtract PHI0 from phi to
            # get the angle wrt the north direction. [deg]
R = 6.371e6 # radius of the earth [m]


def findaAllFiles(folder="InitialData",extn ="/*.nc"):
    """
    Note: find the data file inside a given folder
    """

    mydir = os.getcwd()
    mydir = mydir[0:-len(mydir.split('/')[-1])-1]
    mydir = os.path.join(mydir,folder)
    try:
        os.mkdir(mydir)
        print('New folder added:',mydir)
    except(FileExistsError):
        pass

    files = []
    for file in glob.glob(mydir+extn):
        files.append(file)

    return files


def set_dll_argtypes(dll_swi):
    """
    Fill in the argtypes in the dll object
    so python can call the function correctly
    """
    nx = ctypes.c_size_t
    ny = ctypes.c_size_t
    lat_a = ctypes.c_void_p
    lon_a = ctypes.c_void_p
    height_a = ctypes.c_void_p
    lat = ctypes.c_void_p
    lon = ctypes.c_void_p
    height = ctypes.c_void_p

    dll_swi.regrid_knmi.argtypes = [nx, ny, lat_a, lon_a, height_a, lat, lon, height]
    return dll_swi


def regrid_knmi(lat_a, lon_a, height, lat, lon):
    """
    This is a wrapper for the function of the same name
    in the dll. The wrapper's extra functionality is
    that it finds the dimensions (from the input data)
    and allocates the output array.
    """
    height = np.array(height, dtype=ctypes.c_double)
    retval = np.zeros(lat.shape, dtype=ctypes.c_double)
    tstart = datetime.datetime.now()
    print("starting regrid at t=",tstart)
    dll_swi.regrid_knmi(lat.shape[1], lat.shape[0],
                        lat_a.ctypes.data, lon_a.ctypes.data, height.ctypes.data,
                        lat.ctypes.data, lon.ctypes.data, retval.ctypes.data)
    print("finished regrid at t=",datetime.datetime.now())
    print("regrid took ",datetime.datetime.now()-tstart)
    return retval

def arrays_from_sample_ds(sample_ds):
    """
    Read the contents from a sample_ds object and return them as
    numpy arrays
    """
    height_a = sample_ds.cloud_edge_height.values  * 1e3
    lat      = sample_ds.latitude.values
    lon      = sample_ds.longitude.values
    theta    = sample_ds.theta_view.values
    phi      = sample_ds.phi_view.values

    theta[np.isnan(height_a)] = 0
    phi[np.isnan(height_a)] = 0
    height_a[np.isnan(height_a)] = 0
    return lat, lon, theta, phi, height_a

def cloud_top(sample_ds, name, plot_background=False):
    """
    Create a plot showing the apparent and the actual location of
    the tallest cloud. Optionally, draw the data in the background.
    """
    lat, lon, theta, phi, height_a  = arrays_from_sample_ds(sample_ds)

    # Find the tallest cloud and its properties
    max_h = np.max(height_a)
    w = np.where(height_a==np.max(height_a))
    ix_max = w[0]
    iy_max = w[1]
    lon_max = np.array([lon[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    lat_max = np.array([lat[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    phi_max = np.array([phi[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    theta_max = np.array([theta[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    if plot_background:
        # optionally, plot the data in the background
        plt.contourf(lon,lat,height_a)
    plt.plot(lon_max,lat_max,'*',label=name)

    lon_max = np.mean(lon_max)
    lat_max = np.mean(lat_max)
    phi_max = np.mean(phi_max)
    theta_max = np.mean(theta_max)

    plt.plot(lon_max,lat_max,'r*', label='apparent location')

    # Calculate the actual location of the tallest cloud
    lon_a = lon_max - 180/np.pi * ( max_h * np.sin( (phi_max-PHI0)*np.pi/180 )
                   / (R * np.cos(lat_max*np.pi/180) * np.tan((theta_max-90)*np.pi/180)) )
    lat_a = ( lat_max
              - 180/np.pi *  max_h * np.cos( (phi_max-PHI0)*np.pi/180)
                          / (R * np.tan((theta_max-90)*np.pi/180))
            )
    # Plot a line from the apparent to the actual location
    plt.plot([lon_max, lon_a], [lat_max, lat_a],'r')
    plt.plot(lon_a, lat_a,'ro', label='actual location')


def transform_cloud_edge(sample_ds):
    """
    Return the true positions for the cloud height
    """
    lat, lon, theta, phi, height_a  = arrays_from_sample_ds(sample_ds)

    lon_a = ( lon
              - 180/np.pi * height_a * np.sin( (phi-PHI0)*np.pi/180 )
                   / (R * np.cos(lat*np.pi/180) * np.tan((theta-90)*np.pi/180))
            )
    lat_a = ( lat
              - 180/np.pi *  height_a * np.cos( (phi-PHI0)*np.pi/180)
                          / (R * np.tan((theta-90)*np.pi/180))
            )

    if np.any(np.isnan(lon_a)):
        print("NaNs found in longitude")
        w = np.where(np.isnan(lon_a))
        ix = w[0][0]
        iy = w[1][0]
        print("At (ix,iy) = ",ix,',',iy)
        print("lon_a[",ix,',',iy,']=',lon_a[ix,iy])
        print("height_a[",ix,',',iy,']=',height_a[ix,iy])
        print("phi[",ix,',',iy,']=',phi[ix,iy])
        print("lat[",ix,',',iy,']=',lat[ix,iy])
        print("theta[",ix,',',iy,']=',theta[ix,iy])
        raise RuntimeError("klaar")
    return lat_a, lon_a

def regridded_cloud_edge(sample_ds):
    """
    Return the actual cloud height after regridding it to the
    original (latitude, longitude) grid
    """
    lat, lon, theta, phi, height_a  = arrays_from_sample_ds(sample_ds)

    # Get the (latitude, longitude) grid for the original height field
    lat_a, lon_a = transform_cloud_edge(sample_ds)

    # Regrid the height field to the original (latitude, longitude) grid
    height = regrid_knmi(lat_a, lon_a, height_a, lat, lon)
    return lat, lon, height, height_a

def show_cloud_tops(mydir,show_background=False):
    """
    For two data sets referring to the same time,
    create plots to illustrate the transformation from apparent
    to actual location of the tallest cloud.

    with show_background = True
          you get two figures, each shows one dataset with a line
          connecting the apparent location of the tallest cloud to
          its actual location.
    with show_background = False
          you get one picture, and no background. The picture shows
          two lines, each going from the apparent location to the
          actual location. With a bit of luck, the two actual locations
          will be the same!
    """
    filename1 = os.path.join(mydir,'12_02_10_A3.nc')
    filename2 = os.path.join(mydir,'12_02_10_B1.nc')
    sample_ds = xr.load_dataset(filename1, engine="netcdf4")
    if show_background:
        plt.figure(1)
    cloud_top(sample_ds,filename1, show_background)
    if show_background:
       plt.axis('equal')
       plt.title('Sattellite A3')
       plt.legend()

    sample_ds = xr.load_dataset(filename2, engine="netcdf4")

    if show_background:
       plt.figure(2)
    cloud_top(sample_ds,filename2, show_background)
    if show_background:
       plt.title('Sattellite B1')
    plt.axis('equal')
    plt.legend()

    plt.show()

def show_transform(sample_ds, name):
    """
    Draw two subplots: the top one shows the actual cloud height,
    and the bottom one shows the apparent cloud height
    """
    lat, lon, height, height_a = regridded_cloud_edge(sample_ds)
    plt.subplot(2,1,1)
    plt.contourf(lon,lat,height,levels=30)
    plt.title('actual cloud height '+name)
    plt.axis('equal')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.contourf(lon,lat,height_a,levels=30)
    plt.title('apparent cloud height '+name)
    plt.axis('equal')
    plt.colorbar()


def compare_transform(name1, name2):
    """
    Draw the apparent (bottom) and actual (top) cloud heights for two datasets.

    When you call this for two datasets that refer to the same time,
    the top pictures will hopefully be almost the same.
    """
    plt.figure(1)
    sample_ds = xr.load_dataset(name1, engine="netcdf4")
    show_transform(sample_ds, name1)

    plt.figure(2)
    sample_ds = xr.load_dataset(name2, engine="netcdf4")
    show_transform(sample_ds, name2)
    plt.show()

# Open the dll and prepare it so its functions can be called
if platform.system() == 'Darwin':
    dll_swi = ctypes.cdll.LoadLibrary('./build/libswi_knmi.dylib')
else :
    dll_swi = ctypes.cdll.LoadLibrary('./build/libswi_knmi.so')
dll_swi = set_dll_argtypes(dll_swi)
if __name__ == "__main__":

    folder = "InitialData"
    outputFolder = 'Output'
    if platform.system() == 'Darwin':
        mydir = f"{os.getcwd()}/{folder}"
        outdir = f"{os.getcwd()}/{outputFolder}"
    else :
        mydir = os.getcwd()
        mydir = mydir[0:-len(mydir.split('/')[-1])-1]
        mydir = os.path.join(mydir,folder)
        outdir = os.getcwd()
        outdir = outdir[0:-len(outdir.split('/')[-1])-1]
        outdir = os.path.join(outdir,outputFolder)

    try:
        os.mkdir(mydir)
    except(FileExistsError):
        pass

    try:
        os.mkdir(outdir)
    except(FileExistsError):
        pass

    if True:
        show_cloud_tops(mydir,show_background=False)
        show_cloud_tops(mydir,show_background=True)
    if True:

        compare_transform(os.path.join(mydir,'12_03_10_A2.nc'), os.path.join(mydir,'12_03_10_B2.nc'))
        compare_transform(os.path.join(mydir,'12_02_10_A3.nc'), os.path.join(mydir,'12_02_10_B1.nc'))
        compare_transform(os.path.join(mydir,'12_04_10_A1.nc'), os.path.join(mydir,'12_04_10_B3.nc'))

    for fname in glob.glob(os.path.join(mydir,'12*.nc')):
        sample_ds = xr.load_dataset(fname, engine="netcdf4")
        lat, lon, height, height_a = regridded_cloud_edge(sample_ds)
        my_dict = { 'lat': lat, 'lon': lon, 'height': height}

        fname_nodir = fname.split('/')[-1]
        outfname = os.path.join(outdir,fname_nodir.replace('.nc','.pkl'))

        print("writing pickle file ",outfname)
        with open(outfname,'wb') as file:
            pickle.dump(my_dict, file)
