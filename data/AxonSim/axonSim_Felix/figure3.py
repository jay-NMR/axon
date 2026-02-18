#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:15:00 2023

@author: ysong

to calculate for the big delta of Hansol's data'

to imstall mat73: pip install mat73

set python path to include AxonSim3D

"""

"""
jan 2023. add the code to read the full axon file

"""

from numba import jit, prange, set_num_threads
import numpy as np
import time
import math
from numba import int32, uint8, float32, uint16   # import the types
from numba.experimental import jitclass
import matplotlib.pyplot as plt

import scipy.io

from scipy.optimize import curve_fit

# mat73 to load matlab v7.3 files
import mat73

import AxonSim3D as axon


#% fit two Gaussians

def fit2Gauss(dx,p0,p1,p2,p3):
    return p0 * np.exp(-dx**2/2/p1**2) + p2*np.exp(-dx**2/2/p3**2)

#% plot one direction at once, just to see

# fit the displacement to two Gaussians.

def diff_fit_2Gauss(axonfile,dTimelist,kk, axons, t, n, ldelta=0):
    try:
        print ('reading axon file:', axonfile)
        matfile = mat73.loadmat(axonfile)
        print('use mat73')
        mat_all = matfile.get('axon_all')
        mat = mat_all[kk]
        imsize = np.array(mat.get('image_size'),dtype=np.int32)
        print('Image size:\n', imsize, kk)
       
        xyzres = np.array(mat.get('res_um'))
        print('image res:\n', xyzres)
        
        # read the struct pp.vec
        ddir = mat.get('pp')['vec'].transpose()
        print('ddir:\n',ddir)
        
        pore = int32(np.array(mat.get('pore_voxels')))-1
        print('pore:\n',pore.shape)
       
        image = np.zeros(imsize,dtype=np.uint8)
        for i in range(pore.shape[0]):
            ii,jj,kk = pore[i]
            image[ii,jj,kk]=1
        
    except:
        print ('reading axon file:', axonfile)
        matfile = mat73.loadmat(axonfile)
        print('use mat73')
        mat_all = matfile.get('axon_all')
        mat = mat_all[kk]
        imsize = np.array(mat.get('image_size'),dtype=np.int32)
        print('Image size:\n', imsize, kk)
       
        xyzres = np.array(mat.get('res_um'))
        print('image res:\n', xyzres)
        
        # read the struct pp.vec
        ddir = mat.get('pp')['vec'].transpose()
        print('ddir:\n',ddir)
        
        pore = int32(np.array(mat.get('pore_voxels')))-1
        print('pore:\n',pore.shape)
       
        image = np.zeros(imsize,dtype=np.uint8)
        for i in range(pore.shape[0]):
            ii,jj,kk = pore[i]
            image[ii,jj,kk]=1

    
    

    dclist = np.zeros((dTimelist.shape[0],4))
    i=0
    
    mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/RawData/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
    dx = mat['dx']
    dxdist = mat['dxdist']
    for dTime in dTimelist:
        dx1, dy1, dz1,d0,d1,d2 = axon.bigData(10000, dTime,image,xyzres,ddir,pore,ldelta)
        dtest = d0
        xarray, xbins, _ = plt.hist(dtest,200)
        plt.plot(xbins[0:200],xarray)
        plt.title("1")
        plt.show()
        dtest = dxdist[i]
        plt.plot(dx,dtest)
        plt.title("2")
        plt.show()
        #xarray, xbins, _ = plt.hist(dtest,100)
        #dtest = dtest.reshape(100, 2).sum(axis=1)
        #xarray = dxdist[i]
        #dx = np.linspace(dx[0], dx[-1], 101)
        #xbins = dx
        print(len(xbins))
        print(len(xarray))
        print(len(dx))
        print(len(dxdist))
        
        #xarray, xbins = dxdist[i], dx[:,0]
        #xarray, xbins = plt.hist(dxdist[i],100), dx[:,0]
        lowb = np.sqrt(2*dTime*100)
        upb =  np.sqrt(2*dTime*2000)
        try:
            poptx, _ = curve_fit(fit2Gauss, (xbins[0:-1]+xbins[1:])/2, xarray,bounds=([0,0.01,0,lowb],[np.inf,lowb,np.inf, upb]),maxfev=5000)
            p0,p1,p2,p3 = np.abs(poptx)

            if p1>p3:
                ptmp = p1
                p1=p3
                p3=ptmp
                
                ptmp = p0
                p0=p2
                p2=ptmp
            
            dclist[i] = [p0*p1/(p2*p3+p0*p1),p1**2/2/dTime,p2*p3/(p2*p3+p0*p1),p3**2/2/dTime]
            
            plt.plot(xbins,fit2Gauss(xbins,p0,p1,p2,p3))
            plt.title(', dTime:'+ str(dTime)[0:6]+', DC(um2/s):'+str(int(p3**2/2/dTime)) )
            
            plt.xlabel('displacement, um')
            
    #        print('dTime:', dTime, 'fitting result:',np.floor(p0*p1/(p2*p3+p0*p1)*100),'%,',p1**2/2/dTime,np.floor(p2*p3/(p2*p3+p0*p1)*100),'%,',p3**2/2/dTime)
            print('dTime:%.3g, fitting results:%.3g%%, %.3g, %.3g%%, %.3g' % (dTime,(p0*p1/(p2*p3+p0*p1)*100),p1**2/2/dTime,(p2*p3/(p2*p3+p0*p1)*100),p3**2/2/dTime))
                    

        except:
            dclist[i] =[0,0,0,0]
            print('Curve_fit  error. Did not obtain fitting parameters. %d' % i)
        i +=1
        plt.show()
    return dclist





#%%

#Figure 3
dTimelist= np.array([0.005])#[0.005,0.010,0.015,0.0 20,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1]
imagefilename_mat = '../../big_axon_matrix/axon_geo_all.mat'
dclist = diff_fit_2Gauss(imagefilename_mat,dTimelist,23,4, 0, 10000, ldelta=0)




