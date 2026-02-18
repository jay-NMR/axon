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
    mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/ScaledData/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
    dx = mat['dx'].transpose()
    dxdist = mat['dxdist']
    for dTime in dTimelist:
        #dx1, dy1, dz1,d0,d1,d2 = axon.bigData(50000, dTime,image,xyzres,ddir,pore,ldelta)
    
        #dtest = d0
        
        xarray, xbins = dxdist[i], dx[:,0]
        lowb = np.sqrt(2*dTime*100)
        upb =  np.sqrt(2*dTime*2000)
        try:
            poptx, _ = curve_fit(fit2Gauss, (xbins[0:-1]+xbins[1:])/2, xarray,bounds=([0,0.01,0,lowb],[np.inf,lowb,np.inf, upb]),maxfev=5000)
            print("got here")
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
#Run simulation on specified axons, for 4 different number of particles and number of trials
#If you have no data, run this and direct the savemat line to the correct directory
#Slow, needs to simulate particles
from scipy.io import savemat
imagefilename_mat = '../../big_axon_matrix/axon_geo_all.mat'
nparticles = [1]#[10000, 20000, 40000, 80000]
dataBig = []
varBig = []
for axons in [4]:#[4, 909, 261, 249, 7, 24]
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
                
            dxdist,_, dx, xtrace, ytrace, ztrace =diff_dist_geo(imagefilename_mat,dTimelist,ldelta[1],DC0=DC0,axonStart=axons,axonEnd=axons + 1, NN = n)
            print('Elapsed time: %g min' % ((time.perf_counter()-t0)/60))
            dataBig.append(dxdist)
            plt.title("dx, ldelta-0.01, axon:" + str(axons) + " Trial" + str(t))
            plt.plot(dx,dxdist.transpose())
            plt.ylabel('dist')
            plt.xlabel('displacement, um')
                
                
            plt.show()
            print('All done')
            #mdic = {"axon_name": "Axon " + str(axons), "DTime": dTimelist,"dxdist":dxdist,"ldelta":ldelta,"DC0":DC0,"dx":dx}
           # savemat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/data/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat", mdic)
            #Find the diffusion coefficeint
            
            #finding the mean of x
            array_size = dTimelist.shape[0]
            
            x_mean = np.zeros(array_size)
            xsq_mean = np.zeros(array_size)
            normalize = dxdist[0].sum()
             
            for a in range(array_size):
                for b in range(200):
                    x_mean[a] += (dx[b] * dxdist[a][b])
                    xsq_mean[a] += (dx[b]**2 * dxdist[a][b])
                    #normalize += dxdist[a][b]
                x_mean[a] /= normalize
                xsq_mean[a] /= normalize
            diffdata.append(xsq_mean/2/dTimelist)
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
