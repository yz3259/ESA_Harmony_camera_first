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
        if '_new' in fname:
            continue 
        identification = os.path.basename(fname).replace('.pkl','')
        time = identification[:8]
        camera = identification[9:]
        with open(fname,'rb') as file:
            mydict = pickle.load(file)
        if time in snapshots:
            snapshots[time] = np.maximum(snapshots[time], mydict['height'])
        else:
            snapshots[time] = mydict['height']
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


def find_mainflow(snapshots):
    u = []
    v = []
    times = sorted([key for key in snapshots])
    
    for t0,t1 in zip(times[:-1],times[1:]):
        print(t0,"--",t1)
        # Get the 2D fourier transform of the height now and the height after 1 minute:
        s0 = np.fft.fft2(snapshots[t0])
        s1 = np.fft.fft2(snapshots[t1])
    
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
        u.append(dx)
        v.append(dy)
    return u, v





def shift_snapshots(snapshots, u, v):
    dx = [0]
    dy = [0]
    times = sorted([key for key in snapshots])
    for ui,vi in zip(u,v):
        dx.append(dx[-1]+ui)
        dy.append(dy[-1]+vi)
    
    
    shifted = {}
    for dxi, dyi, time in zip(dx,dy, times):
        dlat = dyi * (lat[1,0]-lat[0,0])
        dlon = dxi * (lon[0,1]-lon[0,0])
        lati = lat + dlat
        loni = lon + dlon
        print("shifting by (%f,%f)" % (dlat,dlon))
        shifted[time] = transform.regrid_knmi(lati, loni, snapshots[time], lat, lon)
        shifted[time][shifted[time] < 0] = 0
    return shifted

def show_shifted(lat,lon, snapshots, shifted):
    # plot the shifted fields to see how well they match:
    times = sorted([key for key in snapshots])
    for i,t0 in enumerate(times):
        plt.figure(i); 
        plt.subplot(2,1,1)
        plt.contourf(lon,lat,snapshots[t0])
        plt.title(t0)
    
        plt.subplot(2,1,2)
        plt.contourf(lon,lat,shifted[t0])
        plt.title(t0+", shifted")
    plt.show()

def show_shiftdiff(lat,lon, snapshots, shifted):
    # plot the shifted fields to see how well they match:
    times = sorted([key for key in snapshots])
    for i,(t0,t1) in enumerate(zip(times[:-1],times[1:])):
        plt.figure(i); 
        plt.contourf(lon,lat,shifted[t0]-shifted[t1])
        plt.title(t0+"-"+t1+", shifted")
        plt.colorbar()
    plt.show()

mydir = os.path.join('..','Output')
lat, lon, snapshots = read_snapshots(mydir)
u,v = find_mainflow(snapshots)
shifted = shift_snapshots(snapshots,u,v)
#show_shiftdiff(lat,lon, snapshots, shifted)

show_shifted(lat,lon, snapshots, shifted)
    
