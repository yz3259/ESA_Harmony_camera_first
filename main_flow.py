#!/usr/bin/env python3
import numpy as np
import os
import dill as pickle
from glob import glob
import matplotlib.pyplot as plt

mydir = os.path.join('..','Output')
 
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

times = sorted([key for key in snapshots])

nlevels = 5
for t0,t1 in zip(times[:-1],times[1:]):
    print(t0,"--",t1)
    # Get the 2D fourier transform of the height now and the height after 1 minute:
    s0 = np.fft.fft2(snapshots[t0])
    s1 = np.fft.fft2(snapshots[t1])

    # Get the inverse 2D fourier transform of the scaled ratio
    ratio = (s0*np.conjugate(s1)) / np.abs(s0*np.conjugate(s1))
    theta = np.angle(ratio)
    v_bulk = theta[:,1:] - theta[:,:-1]
    for i in range(400):
        v_bulk[:,1:] = (v_bulk[:,1:] + v_bulk[:,:-1])/2
        v_bulk[:,0] = v_bulk[:,1]
        v_bulk[:,:-1] = (v_bulk[:,1:] + v_bulk[:,:-1])/2
        v_bulk[:,-1] = v_bulk[:,-2]
        v_bulk[1:,:] = (v_bulk[1:,:] + v_bulk[:-1,:])/2
        v_bulk[0,:] = v_bulk[1,:]
        v_bulk[:-1,:] = (v_bulk[1:,:] + v_bulk[:-1,:])/2
        v_bulk[-1,:] = v_bulk[-2,:]
 
    u_bulk = theta[1:,:] - theta[:-1,:]

    for i in range(400):
        u_bulk[:,1:] = (u_bulk[:,1:] + u_bulk[:,:-1])/2
        u_bulk[:,0] = u_bulk[:,1]
        u_bulk[:,:-1] = (u_bulk[:,1:] + u_bulk[:,:-1])/2
        u_bulk[:,-1] = u_bulk[:,-2]
        u_bulk[1:,:] = (u_bulk[1:,:] + u_bulk[:-1,:])/2
        u_bulk[0,:] = u_bulk[1,:]
        u_bulk[:-1,:] = (u_bulk[1:,:] + u_bulk[:-1,:])/2
        u_bulk[-1,:] = u_bulk[-2,:]

    plt.figure(1)
    plt.contourf(u_bulk)
    plt.colorbar()
    plt.title('u-bulk')

    plt.figure(2)
    plt.contourf(v_bulk)
    plt.colorbar()
    plt.title('v-bulk')
    #plt.show()

    q = np.fft.ifft2(ratio)
    print("   |imag(q)| = ",np.linalg.norm(np.imag(q)))
    print("   |real(q)| = ",np.linalg.norm(np.real(q)))
    # it turns out that q is real: completely remove the imaginary part now
    q = np.real(q)
    plt.contour(q,levels=10)
    #plt.show()

    # The location of q's maximum is the drift vector (in pixels)
    w = np.where(q==np.max(q))

    dx = w[0][0]
    dy = w[1][0]
    dx = dx if dx < q.shape[0]/2 else dx-q.shape[0]
    dy = dy if dy < q.shape[1]/2 else dy-q.shape[1]
    hlp = dx; dx=dy; dy=hlp
    print("dx=",dx,'. dy=',dy)
    #dx,dy=0,0
    # plot the shifted fields to see how well they match:

    # first, in pixels:
    plt.figure(1); plt.clf()
    plt.contourf(np.arange(q.shape[1]),np.arange(q.shape[0]),snapshots[t0])
    plt.xlim(min(dx,0),max(q.shape[1],q.shape[1]+dx))
    plt.ylim(min(dy,0),max(q.shape[0],q.shape[0]+dy))
    plt.clim(0,max_height)
    plt.title(t0)
    plt.figure(2); plt.clf()
    plt.contourf(dx +np.arange(q.shape[1]),dy + np.arange(q.shape[0]),snapshots[t1])
    plt.xlim(min(dx,0),max(q.shape[1],q.shape[1]+dx))
    plt.ylim(min(dy,0),max(q.shape[0],q.shape[0]+dy))
    plt.clim(0,max_height)
    plt.title(t1)
    plt.show()

    plt.figure(1); plt.clf()
    plt.contourf(lon,lat,snapshots[t0])
    plt.xlim(min(dx,0),max(q.shape[1],q.shape[1]+dx))
    plt.ylim(min(dy,0),max(q.shape[0],q.shape[0]+dy))
    plt.clim(0,max_height)
    plt.figure(2); plt.clf()
    plt.contourf(dx +np.arange(q.shape[1]),dy + np.arange(q.shape[0]),snapshots[t1])
    plt.xlim(min(dx,0),max(q.shape[1],q.shape[1]+dx))
    plt.ylim(min(dy,0),max(q.shape[0],q.shape[0]+dy))
    plt.clim(0,max_height)
    plt.show()
    
