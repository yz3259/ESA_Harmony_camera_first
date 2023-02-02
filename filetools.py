import os

def update_xarray(sample_ds,output_tensor,filename,folder="Data"):


    sample_ds["true_latitude"] = 0*sample_ds.latitude.copy(deep=True)
    sample_ds["true_latitude"] += output_tensor[0,:,:]

    sample_ds["true_longitude"] = 0*sample_ds.longitude.copy(deep=True)
    sample_ds["true_longitude"] += output_tensor[1,:,:]


    comp = {"zlib": True, "complevel": 9} # < very small file (just takes a bit of time to write)
    encoding = {var: comp for var in sample_ds.data_vars}

    # write to file
    mypath = os.getcwd()
    mypath = mypath[0:-len(mypath.split('/')[-1])-1]
    mypath = os.path.join(mypath,folder)
    file = filename.split('/')[-1]
    try:
        os.mkdir(mypath)
        print('New folder added:',mypath)
    except(FileExistsError):
        pass
    sample_ds.to_netcdf(mypath+f"/new_{file}", encoding=encoding)
    print(f'{file} is updated to {mypath}/new_{file}!')
