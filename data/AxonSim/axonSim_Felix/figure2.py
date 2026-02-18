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
Sept 9, 2024
This code is for loading previously collected data and looking at the displacement dist of individual axons

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
  
        
        


#Fast
import scipy.io
nparticles = [10000, 20000, 40000, 80000]#[10000, 20000, 40000, 80000]
distBig = []
dataBig = []
varBig = []
aveBig = []
dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1])
for axons in [4, 909, 261, 249, 7, 24]:#[4, 909, 261, 249, 7, 24]
    array = []
    distarray = []
    for n in nparticles:
        diffdata = []
        distdata = []
        for t in range(4):
            #mat = mat73.loadmat(imagefilename_mat)
            #print('use mat73')
            mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/ScaledData/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
            dx = mat['dx'].transpose()
            dxdist = mat['dxdist']
            distdata.append(dxdist)
            plt.title("dx, ldelta-10, axon:" + str(axons) + " Trial" + str(t) + ", pnum: " + str(n))
            plt.plot(dx,dxdist.transpose())
            plt.ylabel('dist')
            plt.xlabel('displacement, um')
            
            
            plt.show()
            #mdic = {"axon_name": "Axon " + str(axons), "DTime": dTimelist,"dxdist":dxdist,"ldelta":ldelta,"DC0":DC0,"dx":dx}
            #savemat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/data/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat", mdic)
            
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
        distarray.append(distdata)
    distBig.append(distarray)
        
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
    aveBig.append(average)



#%%
#Figure 2, displacement distribution
#change range to 6 for all axons
#Normilized to one
pCount = [10000, 20000, 40000, 80000]
for dtime in [0,1,3,9,14]:
    for n in range(4): #Particles count
        for axon in range(6):
            plt.plot(dx, distBig[axon][n][0][dtime]/pCount[n])
        plt.title("Displacement Dist: pnum_" + str(nparticles[n]) + "_dTime=" + str(dTimelist[dtime]))
        plt.xlabel("Displacement, um")
        plt.ylabel("Particle Count")
        plt.legend(["Axon4", "Axon909", "Axon261", "Axon249", "Axon7", "Axon24"])
        plt.show()
    
    
    
    
#%%

import AxonSim3D as axon

def particleTraj(axonfile,dTimelist,ldelta=10,DC0=2000,axonStart=0, axonEnd=-1):
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
    
    
    xtrace = []
    ytrace = []
    ztrace = []
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
    
        volfactor = pore.shape[0]
        i = 0
        for dTime in dTimelist:
            _,_,_,_,_,_,xt, yt, zt = axon.singleTraj(dTime,image,xyzres,ddir,pore,ldelta,DC0=DC0)
            xtrace.append(xt)
            ytrace.append(yt)
            ztrace.append(zt)
            i += 1
    return xtrace, ytrace, ztrace


#Looking at particle tracing
dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1])
imagefilename_mat = '../../big_axon_matrix/axon_geo_all.mat'
DC0=600
ldelta=[0,0.01];
axons = 4

xtrace, ytrace, ztrace = particleTraj(imagefilename_mat,dTimelist,ldelta[0],DC0=DC0,axonStart=axons,axonEnd=axons + 1)

#%%

#Plotting Particle Trajectories, each direction displacement for 100 steps of 1 steps
a = 0
N = 100
size = int(len(xtrace[a])/N)
stepTime = dTimelist[a]/len(xtrace[a]) 
print("each step is " + str(stepTime) + "seconds")
dTime = np.linspace(0, dTimelist[a]*100/len(xtrace[a]), 100)
plt.plot(dTime, xtrace[a][0:N] - xtrace[a][0])
plt.title("Particle first 100 steps, x-position")
plt.xlabel("Time")
plt.ylabel("Displacement in the x")
plt.show()


plt.plot(dTime, ytrace[a][0:N] - ytrace[a][0])
plt.title("Particle first 100 steps, y-position")
plt.xlabel("Time")
plt.ylabel("Displacement in the y")
plt.show()


plt.plot(dTime, ztrace[a][0:N] - ztrace[a][0])
plt.title("Particle first 100 steps, z-position")
plt.xlabel("Time")
plt.ylabel("Displacement in the z")
plt.show()


#%%

#Plotting Particle Trajectories, each direction displacement for 100 steps over total dTime
a = 0
N = 100
size = int(len(xtrace[a])/N)
stepTime = dTimelist[a]/len(xtrace[a]) * size
print("each step is " + str(stepTime) + "seconds")
dTime = np.linspace(0, dTimelist[a], N)
plt.plot(dTime, xtrace[a][::size] - xtrace[a][0])
plt.title("Particle first 100 steps, x-position")
plt.xlabel("Time")
plt.ylabel("Displacement in the x, um")
plt.show()


plt.plot(dTime, ytrace[a][::size] - ytrace[a][0])
plt.title("Particle first 100 steps, y-position, um")
plt.xlabel("Time")
plt.ylabel("Displacement in the y")
plt.show()


plt.plot(dTime, ztrace[a][::size] - ztrace[a][0])
plt.title("Particle first 100 steps, z-position, um")
plt.xlabel("Time")
plt.ylabel("Displacement in the z")
plt.show()

#%%

#3D plots of particles

a = 14
N = 300
size = int(len(xtrace[a])/N)
stepTime = dTimelist[a]/len(xtrace[a]) * size

ax = plt.axes(projection='3d')
xdata = xtrace[a][::size] - xtrace[a][0]
ydata = ytrace[a][::size] - ytrace[a][0]
zdata = ztrace[a][::size] - ztrace[a][0]

ax.plot3D(xdata, ydata, zdata, color='blue', alpha=0.5)
ax.scatter3D(xdata, ydata, zdata, cmap='Greens');



