#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import ctypes
import datetime
import glob
import dill as pickle
import os

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
    lat, lon, theta, phi, height_a  = arrays_from_sample_ds(sample_ds)

    max_h = np.max(height_a)
    w = np.where(height_a==np.max(height_a))
    ix_max = w[0]
    iy_max = w[1]
    lon_max = np.array([lon[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    lat_max = np.array([lat[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    phi_max = np.array([phi[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    theta_max = np.array([theta[ix,iy] for ix,iy in zip(ix_max, iy_max)])
    if plot_background:
        plt.contourf(lon,lat,height_a)
    plt.plot(lon_max,lat_max,'*',label=name)

    lon_max = np.mean(lon_max)
    lat_max = np.mean(lat_max)
    phi_max = np.mean(phi_max)
    theta_max = np.mean(theta_max)

    plt.plot(lon_max,lat_max,'r*', label='apparent location')
    lon_a = lon_max - 180/np.pi * ( max_h * np.sin( (phi_max-PHI0)*np.pi/180 )
                   / (R * np.cos(lat_max*np.pi/180) * np.tan((theta_max-90)*np.pi/180)) )
    lat_a = ( lat_max
              - 180/np.pi *  max_h * np.cos( (phi_max-PHI0)*np.pi/180)
                          / (R * np.tan((theta_max-90)*np.pi/180))
            )
    plt.plot([lon_max, lon_a], [lat_max, lat_a],'r')
    plt.plot(lon_a, lat_a,'ro', label='actual location')


def transform_cloud_edge(sample_ds):
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
    height = regrid_knmi(lat_a, lon_a, height_a, lat, lon)
    return lat, lon, height, height_a

def show_cloud_tops(show_background=False):
    filenames = findaAllFiles(folder="InitialData",extn ="/*.nc")
    filename1 = filenames[9]
    sample_ds = xr.load_dataset(filename1, engine="netcdf4")
    if show_background:
        plt.figure(1)
    cloud_top(sample_ds,'A3', show_background)
    if show_background:
       plt.axis('equal')
       plt.title('Sattellite A3')
       plt.legend()

    filename2 = filenames[5]
    sample_ds = xr.load_dataset(filename2, engine="netcdf4")

    if show_background:
       plt.figure(2)
    cloud_top(sample_ds,'B1', show_background)
    if show_background:
       plt.title('Sattellite B1')
    plt.axis('equal')
    plt.legend()

    plt.show()

def show_transform(sample_ds, name):
    lat, lon, height, height_a = transform_cloud_edge(sample_ds)
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
    plt.figure(1)
    sample_ds = xr.load_dataset(name1, engine="netcdf4")
    show_transform(sample_ds, name1)

    plt.figure(2)
    sample_ds = xr.load_dataset(name2, engine="netcdf4")
    show_transform(sample_ds, name2)
    plt.show()

if __name__ == "__main__":

    dll_swi = ctypes.cdll.LoadLibrary('./build/libswi_knmi.so')
    dll_swi = set_dll_argtypes(dll_swi)

    folder = "InitialData"
    outputFolder = 'Output'
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
        show_cloud_tops(show_background=False)
        show_cloud_tops(show_background=True)
    if True:

        compare_transform(os.path.join(mydir,'12_03_10_A2.nc'), os.path.join(mydir,'12_03_10_B2.nc'))
        compare_transform(os.path.join(mydir,'12_02_10_A3.nc'), os.path.join(mydir,'12_02_10_B1.nc'))
        compare_transform(os.path.join(mydir,'12_04_10_A1.nc'), os.path.join(mydir,'12_04_10_B3.nc'))

    for fname in glob.glob(os.path.join(mydir,'12*.nc')):
        sample_ds = xr.load_dataset(fname, engine="netcdf4")
        lat, lon, height, height_a = transform_cloud_edge(sample_ds)
        my_dict = { 'lat': lat, 'lon': lon, 'height': height}


        fname_nodir = fname.split('/')[-1]
        outfname = os.path.join(outdir,fname_nodir.replace('.nc','.pkl'))

        print("writing pickle file ",outfname)
        with open(outfname,'wb') as file:
            pickle.dump(my_dict, file)
    quit()
