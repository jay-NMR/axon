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



# nov 16
def diff_dist(axonfile,dTimelist,ldelta=0,DC0=2000):
    try:
        mat = scipy.io.loadmat(axonfile) 
        print('use scipy.io.loadmat')
        image = np.array(mat.get('BB'))
        #print(image[500,500])
        # smallImage = image[150:550, 50:450, 350:750]
        print('Image size:\n', image.shape)
        
        xyzres = np.array(mat.get('res_um'))[0]
        print('image res:\n', xyzres)
        
        # read the struct pp.vec
        ddir = np.array(mat.get('pp'))['vec'][0][0].transpose()
        print('ddir:\n',ddir)
        
        pore = int32(np.array(mat.get('pore_voxels')))-1
        print('pore:\n',pore.shape)
        
        
    except:
        mat = mat73.loadmat(axonfile)
        print('use mat73')
        
        #image = np.array(mat.get('BB'))
        #print(image[500,500])
        # smallImage = image[150:550, 50:450, 350:750]        
        imsize = np.array(mat.get('image_size'),dtype=np.int32)
        print('Image size:\n', imsize)
       
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
    
    dxlist = np.zeros((dTimelist.shape[0],200))
    dx = np.linspace(-25,25,201)
    dx1 = (dx[0:-1]+dx[1:])/2
    
    NN = 10000
    volfactor = pore.shape[0]/NN
    i=0
    for dTime in dTimelist:
        _, _, _,d0,d1,d2 = axon.bigData(NN, dTime,image,xyzres,ddir,pore,ldelta,DC0=DC0)
        
    
        dtest = d0
        xarray, xbins, _ = plt.hist(dtest,dx)
    
        dxlist[i] = xarray*volfactor
        
        i +=1
        plt.show()
    
    return dxlist, dx, dx1


# jan 21, 2023. diff_distr from axon_geo_all.mat, which contains the largest 3500 axons
def diff_dist_geo(axonfile,dTimelist,ldelta=10,DC0=2000,axonStart=0, axonEnd=-1, NN=10000):

    print ('reading axon file:', axonfile)
    matfile = mat73.loadmat(axonfile)
    print('use mat73')
    mat_all = matfile.get('axon_all')
    
    dxlist = np.zeros((dTimelist.shape[0],200))
    dx = np.linspace(-25,25,201)
    dx1 = (dx[0:-1]+dx[1:])/2
    
    #NN = 10000
    # pick an axon to run sim
    if axonEnd<0:
        axonEnd = len(mat_all)
        
    print(axonfile)
    print('number of axons:', axonStart, '-', axonEnd)
    
    
    
    for kk in range(axonStart,axonEnd):
    #for kk in range(0,100):    
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
    
        volfactor = pore.shape[0]/NN
        i=0
        for dTime in dTimelist:
            _, _, _,d0,d1,d2, xtrace, ytrace, ztrace = axon.bigData(NN, dTime,image,xyzres,ddir,pore,ldelta,DC0=DC0)

            dtest = d0
            xarray, xbins, _ = plt.hist(dtest,dx)
        
            dxlist[i] += xarray * volfactor
            i += 1
        
    return dxlist, dx, dx1, xtrace, ytrace, ztrace


# Aug 24, 2023. diff_distr from axon_geo_all.mat, which contains the largest 3500 axons
#Looking at orthogonal directions
def diff_dist_geo_orth(axonfile,dTimelist,ldelta=10,DC0=2000,axonStart=0, axonEnd=-1, NN=10000):

    print ('reading axon file:', axonfile)
    matfile = mat73.loadmat(axonfile)
    print('use mat73')
    mat_all = matfile.get('axon_all')
    
    dylist = np.zeros((dTimelist.shape[0],200))
    dy = np.linspace(-25,25,201)
    dy1 = (dy[0:-1]+dy[1:])/2
    
    dzlist = np.zeros((dTimelist.shape[0],200))
    dz = np.linspace(-25,25,201)
    dz1 = (dz[0:-1]+dz[1:])/2
    
    # pick an axon to run sim
    if axonEnd<0:
        axonEnd = len(mat_all)
        
    print(axonfile)
    print('number of axons:', axonStart, '-', axonEnd)
    
    
    
    for kk in range(axonStart,axonEnd):
    #for kk in range(0,100):    
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
    
        volfactor = pore.shape[0]/NN
        i=0
        for dTime in dTimelist:
            _, _, _,d0,d1,d2 = axon.bigData(NN, dTime,image,xyzres,ddir,pore,ldelta,DC0=DC0)
        
            dtesty = d1
            yarray, ybins, _ = plt.hist(dtesty,dy)
            dtestz = d2
            zarray, zbins, _ = plt.hist(dtestz,dz)
        
            dylist[i] += yarray*volfactor
            dzlist[i] += zarray*volfactor
            i += 1
        
        
    return dylist, dy, dy1, dzlist, dz, dz1


#%%
#Running simulation and looking at transverse directions
from scipy.io import savemat
imagefilename_mat = '../../big_axon_matrix/axon_geo_all.mat'
nparticles = [10000, 20000, 40000, 80000]#[10000, 20000, 40000, 80000]
dataBig = []
varBig = []
for axons in [4, 909, 261, 249, 7, 24]:#[4, 909, 261, 249, 7, 24]
    array = []
    for n in nparticles:
        diffdata = []
        for t in range(4):
            #mat = mat73.loadmat(imagefilename_mat)
            #print('use mat73')
            t0 = time.perf_counter()
            
            dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1]) # use 19
            #dTimelist= np.array([0.005])
            DC0=600
            ldelta=[0,0.01];
            
            dydist,_, dy, dzdist,_, dz=diff_dist_geo_orth(imagefilename_mat,dTimelist,ldelta[0],DC0=DC0,axonStart=axons,axonEnd=axons+1, NN=n)
            print('Elapsed time: %g min' % ((time.perf_counter()-t0)/60))

            plt.title("dy, ldelta-10, axon:" + str(axons) + " Trial" + str(t) + ", pnum: " + str(n))
            plt.plot(dy,dydist.transpose())
            plt.ylabel('dist - dy')
            plt.xlabel('displacement, um')
            plt.show()

            plt.title("dz, ldelta-10, axon:" + str(axons) + " Trial" + str(t) + ", pnum: " + str(n))
            plt.plot(dz,dzdist.transpose())
            plt.ylabel('dist - dz')
            plt.xlabel('displacement, um')     
            plt.show()
            print('All done')
            
           # mdic = {"axon_name": "Axon " + str(axons), "DTime": dTimelist,"dydist":dydist, "dzdist":dzdist, "ldelta":ldelta,"DC0":DC0,"dy":dy,"dz":dz}
           # savemat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/data/axon" + str(axons) + "/dyzdist_trial" + str(t) + "_pnum" + str(n) + ".mat", mdic)
#%%
      
#Reading the tranverse direction data - y direction
import scipy.io
nparticles = [10000, 20000, 40000, 80000]#[10000, 20000, 40000, 80000]
distBig = []
dataBig = []
varBig = []
aveBig = []
for axons in [4, 909, 261, 249, 7, 24]:#[4, 909, 261, 249, 7, 24]
    array = []
    distarray = []
    for n in nparticles:
        diffdata = []
        distdata = []
        for t in range(4):
            #mat = mat73.loadmat(imagefilename_mat)
            #print('use mat73')
            mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/travData/axon" + str(axons) + "/dyzdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
            dy = mat['dy'].transpose()
            dydist = mat['dydist']
            dz = mat['dz'].transpose()
            dzdist = mat['dzdist']
            
            dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1])
            distdata.append(dydist)
            plt.title("dy, ldelta-10, axon:" + str(axons) + " Trial" + str(t) + ", pnum: " + str(n))
            plt.plot(dy,dydist.transpose())
            plt.ylabel('dist')
            plt.xlabel('displacement, um')
            
            
            plt.show()
            #Find the diffusion coefficeint
            
            #finding the mean of x
            array_size = dTimelist.shape[0]
            
            y_mean = np.zeros(array_size)
            ysq_mean = np.zeros(array_size)
            normalize = dydist[0].sum()
             
            for a in range(array_size):
                for b in range(200):
                    y_mean[a] += (dy[b] * dydist[a][b])
                    ysq_mean[a] += (dy[b]**2 * dydist[a][b])
                    #normalize += dxdist[a][b]
                y_mean[a] /= normalize
                ysq_mean[a] /= normalize
            diffdata.append(ysq_mean/2/dTimelist)
        array.append(diffdata)
        
    for num in range(len(nparticles)):
        plt.plot(dTimelist, array[num][0])
        plt.plot(dTimelist, array[num][1])
        plt.plot(dTimelist, array[num][2])
        plt.plot(dTimelist, array[num][3])
        plt.title("diffusion coefficient," + str(nparticles[num]) + "Particles, axon-" + str(axons))
        plt.xlabel("diffusion time - s")
        plt.ylabel("Diffusion coefficient - um^2/s")
        plt.show()
        
    dataBig.append(array)
    average = []
    #Calculate variance
    #Calculate diffusion constant at 40000
    for Time in range(array_size):
        ave = 0
        for data in range(4):
            ave += array[3][data][Time]
        ave /= 4
        average.append(ave)
    varData = []
    for num in [0,1,2,3]:
        variance = []
        for Time in range(array_size):
            var = 0
            for data in range(4):
                var += (array[num][data][Time] - average[Time])**2
            var /= 4
            var = math.sqrt(var)
            variance.append(var)
        varData.append(variance)
        plt.plot(dTimelist, variance)
        plt.title("variance calculation," + str(nparticles[num]) + "particles, axons-" + str(axons))
        plt.ylim(0,15)
        plt.xlabel("dTime")
        plt.ylabel("Variance")
        plt.show()
    varBig.append(varData)
        


#%%
#In the z direction

import scipy.io
nparticles = [10000, 20000, 40000, 80000]#[10000, 20000, 40000, 80000]
distBig = []
dataBig = []
varBig = []
aveBig = []
for axons in [4, 909, 261, 249, 7, 24]:#[4, 909, 261, 249, 7, 24]
    array = []
    distarray = []
    for n in nparticles:
        diffdata = []
        distdata = []
        for t in range(4):
            #mat = mat73.loadmat(imagefilename_mat)
            #print('use mat73')
            mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/travData/axon" + str(axons) + "/dyzdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
            dz = mat['dz'].transpose()
            dzdist = mat['dzdist']
            
            dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1])
            distdata.append(dzdist)
            plt.title("dz, ldelta-10, axon:" + str(axons) + " Trial" + str(t) + ", pnum: " + str(n))
            plt.plot(dy,dydist.transpose())
            plt.ylabel('dist')
            plt.xlabel('displacement, um')
            
            
            plt.show()
            #Find the diffusion coefficeint
            
            #finding the mean of x
            array_size = dTimelist.shape[0]
            
            z_mean = np.zeros(array_size)
            zsq_mean = np.zeros(array_size)
            normalize = dzdist[0].sum()
             
            for a in range(array_size):
                for b in range(200):
                    z_mean[a] += (dz[b] * dzdist[a][b])
                    zsq_mean[a] += (dz[b]**2 * dzdist[a][b])
                    #normalize += dxdist[a][b]
                z_mean[a] /= normalize
                zsq_mean[a] /= normalize
            diffdata.append(zsq_mean/2/dTimelist)
        array.append(diffdata)
        
    for num in range(len(nparticles)):
        plt.plot(dTimelist, array[num][0])
        plt.plot(dTimelist, array[num][1])
        plt.plot(dTimelist, array[num][2])
        plt.plot(dTimelist, array[num][3])
        plt.title("diffusion coefficient," + str(nparticles[num]) + "Particles, axon-" + str(axons))
        plt.xlabel("diffusion time - s")
        plt.ylabel("Diffusion coefficient - um^2/s")
        plt.show()
        
    dataBig.append(array)
    average = []
    #Calculate variance
    #Calculate diffusion constant at 40000
    for Time in range(array_size):
        ave = 0
        for data in range(4):
            ave += array[3][data][Time]
        ave /= 4
        average.append(ave)
    varData = []
    for num in [0,1,2,3]:
        variance = []
        for Time in range(array_size):
            var = 0
            for data in range(4):
                var += (array[num][data][Time] - average[Time])**2
            var /= 4
            var = math.sqrt(var)
            variance.append(var)
        varData.append(variance)
        plt.plot(dTimelist, variance)
        plt.title("variance calculation," + str(nparticles[num]) + "particles, axons-" + str(axons))
        plt.ylim(0,15)
        plt.xlabel("dTime")
        plt.ylabel("Variance")
        plt.show()
    varBig.append(varData)
        