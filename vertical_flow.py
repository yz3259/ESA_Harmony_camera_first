#!/usr/bin/env python3
import numpy as np
import os
import dill as pickle
from glob import glob
import matplotlib.pyplot as plt
import main_flow
import transform

IGNORE_HEIGHTS_BELOW = 500
DISPLACEMENT_TOO_MUCH = 100

def neighbors(dh):
    ngh = [None for i in range(4)]
    ngh[0] = dh.copy(); ngh[0][:,1:]  = ngh[0][:,:-1]
    ngh[1] = dh.copy(); ngh[1][:,:-1] = ngh[1][:,1:] 
    ngh[2] = dh.copy(); ngh[2][:-1,:] = ngh[2][1:,:]  
    ngh[3] = dh.copy(); ngh[3][1:,:]  = ngh[3][:-1,:]   
    return ngh

def smooth_zflow(dh, nsmooth):
    for i in range(nsmooth):

        dh_new = np.zeros(dh.shape)
        nk     = np.zeros(dh.shape)

        for ngh in neighbors(dh):

            isok = np.logical_not(np.isnan(ngh))
            nk += isok
            dh_new[isok] += ngh[isok]

        dh = dh_new.copy() / nk

    return dh

def find_zflow(snapshots):
    times = sorted([key for key in snapshots])
    zflow = {}
    for t0,t1 in zip(times[:-1],times[1:]):
        print(t0,"--",t1)
        # Get the 2D fourier transform of the height now and the height after 1 minute:
        dh = snapshots[t1] -  snapshots[t0]
        dh[snapshots[t1]<=IGNORE_HEIGHTS_BELOW] = np.nan
        dh[snapshots[t0]<=IGNORE_HEIGHTS_BELOW] = np.nan
        dh = np.minimum(DISPLACEMENT_TOO_MUCH,dh)
        dh = np.maximum(-DISPLACEMENT_TOO_MUCH,dh)

        dh = smooth_zflow(dh, nsmooth=40)

        isblank = np.isnan(dh)
        isok = np.logical_not(isblank)
        print("The mean vertical flow is", np.mean(dh[isok]) )
        dh[isblank] = np.mean(dh[isok])

        for it in range(100):
            dh_new = np.zeros(dh.shape)
            for ngh in neighbors(dh):
                dh_new[isblank] += ngh[isblank]
            dh[isblank] = dh_new[isblank]/4

        zflow[t0+" : "+t1] = dh

    return zflow

def shift_zflow(zflow,snapshots):
    times = sorted([key for key in snapshots])
    cumulative = None
    retval = {}
    retval[times[0]] = snapshots[times[0]].copy()

    for t0,t1 in zip(times[:-1],times[1:]):
        print(t0,"--",t1)
        if cumulative is None:
            cumulative = zflow[t0+" : "+t1].copy()
        else:
            cumulative += zflow[t0+" : "+t1]
        retval[t1] = snapshots[t1] - cumulative

    return retval

def show_zflow(lat,lon, zflow):
    # plot the shifted fields to see how well they match:
    times = sorted([key for key in zflow])
    for i,t0 in enumerate(times):
        plt.figure(i);
        plt.contourf(lon,lat,zflow[t0])
        plt.title('dh/dz '+t0)
        plt.colorbar()
    plt.show()

mydir = os.path.join('..','Output')
lat, lon, snapshots = main_flow.read_snapshots(mydir)
u,v = main_flow.find_mainflow(snapshots)
shifted = main_flow.shift_snapshots(lat,lon,snapshots,u,v)
zflow = find_zflow(shifted)
z_shifted = shift_zflow(zflow,shifted)

show_zflow(lat,lon, zflow)

main_flow.show_shifted(lat,lon, snapshots, z_shifted)
main_flow.show_shifted(lat,lon, shifted, z_shifted)

