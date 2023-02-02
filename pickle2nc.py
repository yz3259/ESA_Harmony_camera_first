import dill as pickle
import xarray as xr
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import filetools as ft

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

def findPath(folder="InitialData"):
    """
    Note: find the data file inside a given folder
    """

    mydir = os.getcwd()
    mydir = mydir[0:-len(mydir.split('/')[-1])-1]
    mydir = os.path.join(mydir,folder)

    return mydir






def find_paired_Data(infname,folder = "InitialData"):
    mypath = findPath(folder=folder)
    infname_str = infname.split('/')[-1]
    print('split one',infname_str)
    if "new" in infname_str:
        pair_fname = glob.glob(mypath+f"/*{infname_str[0:-8]}.nc")
        print(f"/{infname_str[0:-8]}.nc")
        print(pair_fname)
    else:
        pair_fname = glob.glob(mypath+f"/*{infname_str}")

    return pair_fname[0]


def extract_tensor(sample_ds):
    # extract date to numpy arrays
    heights = sample_ds.cloud_edge_height.values
    lats = sample_ds.latitude.values
    lons = sample_ds.longitude.values
    theta = sample_ds.theta_view.values
    phi = sample_ds.phi_view.values

    print(heights.shape)
    print(lats.shape)
    print(lons.shape)
    print(theta.shape)
    print(phi.shape)

    # from bottom to top lat, lons, heights, theta, phi
    input_tensor = np.stack([lats,lons,heights,theta,phi], axis=0)
    print('shape of input tensor: ',input_tensor.shape)
    return input_tensor


def scatterPlot(output_tensor,input_tensor,filename):
    namelists = filename.split('/')
    print(namelists)
    lons = output_tensor[1,:,:].flatten()         # 1D numpy
    lats = output_tensor[0,:,:].flatten()          # 1D numpy
    cths = input_tensor[2,:,:].flatten() # 1D numpy

    ## Then, you can e.g. simply do (nb: points plotted individually):
    plt.figure(figsize=(15,10))
    plt.scatter(lons, lats, c=cths, cmap='magma', s=0.1)
    plt.title(namelists[-1])
    plt

if __name__ == "__main__":


    filenames = findaAllFiles(folder="Output",extn ="/*.pkl")
    print(filenames)
    filenames_x = findaAllFiles(folder="InitialData",extn ="/*.nc")
    # print(filenames_x)

    filenumber = int(input('Give a digit between 0 and 9 to load a file.'))
    print(filenames[filenumber])
    infname = filenames[filenumber]
    if "true" in infname:

        ncfname = find_paired_Data(infname,folder = "InitialData")
        print('nc filename: ',ncfname)
        with open(infname,'rb') as file:
            data = pickle.load(file)
            print(data)

        height = data["height"]
        true_lon = data["lon"]
        true_lat = data["lat"]

        output_tensor = np.stack([true_lat,true_lon],axis=0)

        sample_ds = xr.load_dataset(ncfname, engine="netcdf4")
        ft.update_xarray(sample_ds,output_tensor,ncfname,folder="Data")

        # plt.figure(figsize=(16,10))
        # plt.contourf(lon,lat,height,levels=30)
        # plt.title(infname.split('/')[-1])
        # plt.savefig(f"{infname.split('/')[-1]}.png",dpi=300)


    filename = find_paired_Data(infname,folder = "Data")
    sample_ds = xr.load_dataset(filename, engine="netcdf4")
    # NB: - open_dataset opens lazily and turns the .nc file to read-only until the corresp. ds has been .closed().
    #     - load_dataset loads all the .nc contents into memory closes the .nc file immediately after.

    # info for that xr.Dataset:
    display(sample_ds)
    true_lat = sample_ds.true_latitude.values
    true_lon = sample_ds.true_longitude.values
    data = np.stack([true_lat,true_lon],axis=0)
    data.shape
    input_tensor = extract_tensor(sample_ds)

    print('this is stored data')
    scatterPlot(data,input_tensor,filename)


    # rename .nc files
    if False:

        mypath = findPath(folder='Data')
        title = 'true_position'
        for filename in os.listdir(mypath):
            print(filename[4:])
            print(filename.split('_'))
            for string in filename.split('_'):
                try:
                    int(string)
                    idx = filename.find(string)
                  #  print(idx)
                    break
                except ValueError:
                    continue

            dst = f"{title}_{filename[idx:]}"
            os.rename(os.path.join(mypath,filename),os.path.join(mypath,dst))


    # rename .pkl files
    if False:
        mypath = findPath(folder='Output')
        title2 = 'regridded_position'
        title1 = 'true_latlon'
        for filename in os.listdir(mypath):

            print(filename.split('_'))
            for string in filename.split('_'):
                try:
                    int(string)
                    idx = filename.find(string)
                  #  print(idx)
                    break
                except ValueError:
                    continue

            if 'new' in filename:
                print(filename.split('_'))
                idx2 = filename.find('new')
                dst = f"{title1}_{filename[idx:idx2-1]}.pkl"
                print('dst: ',dst)
            else:
                dst = f"{title2}_{filename[idx:]}"
                print('dst: ',dst)

            os.rename(os.path.join(mypath,filename),os.path.join(mypath,dst))
