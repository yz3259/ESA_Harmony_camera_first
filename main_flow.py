#!/usr/bin/env python3
import numpy as np
import os
import dill as pickle
from glob import glob
import matplotlib.pyplot as plt
import transform

def read_snapshots(mydir):
    snapshots = {}
    lat, lon = None, None
    max_height = 0
    for fname in glob(os.path.join(mydir,'*.pkl')):
        print("file",fname)
        if 'regrid' not in fname:
            continue
        identification = os.path.basename(fname).replace('.pkl','')
        print(identification.split('_'))

        # find the time string in file name
        for string in identification.split('_'):
            try:
                int(string)
                idx = identification.find(string)
                break
            except ValueError:
                continue

        time = identification[idx:-3]
        print('time:',time)
        if time in ['12_02_10', '12_03_10', '12_04_10', '12_05_10', '12_06_10']:
            continue
        with open(fname,'rb') as file:
            mydict = pickle.load(file)
        if time in snapshots:
            snapshots[time] = np.maximum(snapshots[time], mydict['height'])
        else:
            snapshots[time] = mydict['height']
        snapshots[time][snapshots[time]<0] = 0
        if lat is not None:
            if np.any(lat != mydict['lat']):
                raise RuntimeError('differences in lat')
            if np.any(lon != mydict['lon']):
                raise RuntimeError('differences in lon')
        else:
            lat = mydict['lat']
            lon = mydict['lon']
        max_height = max(max_height,np.max(mydict['height']))
    return lat, lon, snapshots

def snapshots_2_mainflow(snap0, snap1):
    # Get the 2D fourier transform of the height now and the height after 1 minute:
    s0 = np.fft.fft2(snap0)
    s1 = np.fft.fft2(snap1)

    # Get the inverse 2D fourier transform of the scaled ratio
    ratio = (s0*np.conjugate(s1)) / np.abs(s0*np.conjugate(s1))
    q = np.fft.ifft2(ratio)
    print("   |imag(q)| = ",np.linalg.norm(np.imag(q)))
    print("   |real(q)| = ",np.linalg.norm(np.real(q)))

    # it turns out that q is real: completely remove the imaginary part now
    q = np.real(q)

    # The location of q's maximum is the drift vector (in pixels)
    w = np.where(q==np.max(q))

    dx = w[1][0]
    dy = w[0][0]
    dx = dx if dx < q.shape[1]/2 else dx-q.shape[1]
    dy = dy if dy < q.shape[0]/2 else dy-q.shape[0]
    return dx, dy

def window_function(shape, ixy_start, ixy_end, ntaper):
    retval = np.zeros(shape)
    retval[ixy_start[0], ixy_start[1]:ixy_end[1]] = 1
    retval[ixy_start[0]:ixy_end[0], ixy_start[1]] = 1

    ntap = min(ixy_start[0]-1,ntaper)
    ix_min = ixy_start[0]-ntap 
    for itap in range(ntap):
        retval[ixy_start[0]-ntap+itap, ixy_start[1]] = (1-np.cos( (itap/ntap) * np.pi )) /2
    ntap = min(shape[0]-ixy_end[0]-1,ntaper)
    ix_max = ixy_end[0]+ntap 
    for itap in range(ntap):
        retval[ixy_end[0]+ntap-itap-1, ixy_start[1]] = (1-np.cos( (itap/ntap) * np.pi )) /2
    ntap = min(ixy_start[1]-1,ntaper)
    iy_min = ixy_start[1]-ntap 
    for itap in range(ntap):
        retval[ixy_start[0], ixy_start[1]-ntap+itap] = (1-np.cos( (itap/ntap) * np.pi )) /2
    ntap = min(shape[1]-ixy_end[1]-1,ntaper)
    iy_max = ixy_end[1]+ntap 
    for itap in range(ntap):
        retval[ixy_start[0],ixy_end[1]+ntap-itap-1] = (1-np.cos( (itap/ntap) * np.pi )) /2
    for ix in range(ix_min, ix_max+1):
        retval[ix,iy_min:iy_max+1] = retval[ixy_start[0],iy_min:iy_max+1] * retval[ix, ixy_start[1]] 
    return retval

def find_mainflow(snapshots, mwindow = 1, u=None, v=None):
    times = sorted([key for key in snapshots])
    shape = snapshots[times[0]].shape

    if u is None:
        u = {}
    if v is None:
        v = {}

    for t0,t1 in zip(times[:-1],times[1:]):
        print(t0,"--",t1)

        # Get the 2D fourier transform of the height now and the height after 1 minute:
        vel_u  = np.zeros(shape)
        vel_v  = np.zeros(shape)
        weight = np.zeros(shape)
 
        for iwindow in range(mwindow):
            for jwindow in range(mwindow):
                ixy_start = [int(iwindow*shape[0]/mwindow), int(jwindow*shape[1]/mwindow)]
                ixy_end   = [int((iwindow+1)*shape[0]/mwindow), int((jwindow+1)*shape[1]/mwindow)]
                ntaper    = int(0.3*(shape[0] + shape[1])/(2*mwindow))
                w = window_function(shape, ixy_start = ixy_start, ixy_end = ixy_end, ntaper = ntaper)
                if False:
                    plt.contourf(w)
                    plt.colorbar()
                    plt.show()
                dx, dy = snapshots_2_mainflow(w*snapshots[t0], w*snapshots[t1])
                print("************** found dx,dy=",dx,dy)
                vel_u  += dx*w
                vel_v  += dy*w
                weight += w
        
        t0t1 = t0 + ' : ' + t1 
        if t0t1 in u:
           u[t0t1] += vel_u/weight
        else:
           u[t0t1] = vel_u/weight
        if t0t1 in v:
           v[t0t1] += vel_v/weight
        else:
           v[t0t1] = vel_v/weight

    return u, v


def shift_snapshots(lat,lon,snapshots, u, v):
    times = sorted([key for key in snapshots])
    dx,dy={},{}
    dx[times[0]] = np.zeros(snapshots[times[0]].shape)
    dy[times[0]] = np.zeros(snapshots[times[0]].shape)

    for t0,t1 in zip(times[:-1],times[1:]):
        t0t1 = t0 + ' : ' + t1 
        dx[t1] = dx[t0] + u[t0t1]
        dy[t1] = dy[t0] + v[t0t1]

    shifted = {}
    for time in times:
        dlat = dy[time] * (lat[1,0]-lat[0,0])
        dlon = dx[time] * (lon[0,1]-lon[0,0])
        lati = lat + dlat
        loni = lon + dlon
        shifted[time] = transform.regrid_knmi(lati, loni, snapshots[time], lat, lon)
        shifted[time][shifted[time] < 0] = 0
    return shifted

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

def show_shifted(lat,lon, snapshots, shifted):

    image_dir = findPath(folder="bulk_image")
    # plot the shifted fields to see how well they match:
    times = sorted([key for key in snapshots])
    for i,t0 in enumerate(times):
        plt.figure(i,figsize=(16, 16));
        plt.subplot(2,1,1)
        plt.contourf(lon,lat,snapshots[t0])
        plt.xlim(300.85,300.94)
        plt.ylim(13.795,13.860)
        plt.title(t0)

        plt.subplot(2,1,2)
        plt.contourf(lon,lat,shifted[t0])
        plt.xlim(300.85,300.94)
        plt.ylim(13.795,13.860)
        plt.title(t0+", shifted")
        plt.savefig(os.path.join(image_dir,f'paired_{t0}.png'),dpi=300)
    plt.show()

    for i,t0 in enumerate(times):
        plt.figure(i,figsize=(16, 10));
        plt.subplot(1,1,1)
        plt.contourf(lon,lat,shifted[t0])
        plt.title(t0+", shifted")
        plt.savefig(os.path.join(image_dir,f'bulk_motion_{t0}.png'),dpi=300)
    plt.show()


def show_advec(lat,lon, u,v):

    image_dir = findPath(folder="bulk_image")
    # plot the shifted fields to see how well they match:
    for i,t0t1 in enumerate(sorted([key for key in u])):
        plt.figure(i,figsize=(16, 16));
        plt.subplot(2,1,1)
        plt.contourf(lon,lat,u[t0t1])
        plt.colorbar()
        plt.title('u at '+t0t1)

        plt.subplot(2,1,2)
        plt.contourf(lon,lat,v[t0t1])
        plt.colorbar()
        plt.title('v at '+t0t1)
    plt.show()



def show_shiftdiff(lat,lon, snapshots, shifted):
    # plot the shifted fields to see how well they match:
    times = sorted([key for key in snapshots])
    for i,(t0,t1) in enumerate(zip(times[:-1],times[1:])):
        plt.figure(i);
        plt.contourf(lon,lat,shifted[t0]-shifted[t1])
        plt.title(t0+"-"+t1+", shifted")
        plt.colorbar()
        plt.savefig(f'{t0}_{t1}.png',dpi=300)
    plt.show()


if __name__ == "__main__":
    if False:
        test = window_function(shape=[1000,600], ixy_start = [900,500], ixy_end = [1000,600], ntaper=30)
        plt.contourf(test)
        plt.show()
        test = window_function(shape=[1000,600], ixy_start = [100,100], ixy_end = [200,300], ntaper=30)
        plt.contourf(test)
        plt.show()
        test = window_function(shape=[1000,600], ixy_start = [0,0], ixy_end = [100,200], ntaper=10)
        plt.contourf(test)
        plt.show()
        quit()
    mydir = os.path.join('..','Output')
    lat, lon, snapshots = read_snapshots(mydir)

    u,v = find_mainflow(snapshots)
    shifted0 = shift_snapshots(lat,lon,snapshots,u,v)
    u,v = find_mainflow(shifted0, mwindow = 2, u=u, v=v)
    shifted = shift_snapshots(lat,lon,snapshots,u,v)
    u,v = find_mainflow(shifted, mwindow = 4, u=u, v=v)
    shifted = shift_snapshots(lat,lon,snapshots,u,v)
    u,v = find_mainflow(shifted, mwindow = 8, u=u, v=v)
    shifted = shift_snapshots(lat,lon,snapshots,u,v)

    show_advec(lat,lon, u,v)
    show_shifted(lat,lon, shifted0, shifted)
    show_shifted(lat,lon, snapshots, shifted)
