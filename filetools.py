import os

def update_xarray(sample_ds,output_tensor,filename,folder="Data",title = 'true_position'):


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
    sample_ds.to_netcdf(mypath+f"/{title}_{file}", encoding=encoding)
    print(f'{file} is updated to {mypath}/{title}_{file}!')


def findPath(folder="InitialData"):
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

    return mydir


def binarylise_image(image,filter_height):
    """
    turn image into binary image with a height filter
    """
     # mask out clouds below 1.5 km
    idx = image < filter_height
    image_b = image.copy()
    image_b[idx]=0
    idx = image>filter_height
    image_b[idx]=1
    return image_b

def match_unmatch_mask(image1,image2,filter_height):
    """
    Return: matched points mask, unmatched pts mask
    """
    # binary mask for each image
    image1_b = binarylise_image(image1,filter_height)
    image2_b = binarylise_image(image2,filter_height)

    # matched pts:
    S_c = image1_b * image2_b
    # unmatched pts:
    S = abs(image1_b - image2_b)

    return S_c, S


def matched_pts(image1,image2,filter_height):
    """
    Return the matched points set of two images
    """
    S_c,_ = match_unmatch_mask(image1,image2,filter_height)
    return image1*S_c


def unmatched_pts(image1,image2,filter_height):
    """
    Return unmatched points of image1 then those of image 2
    """
    _,S = match_unmatch_mask(image1,image2,filter_height)
    return image1*S,image2*S
