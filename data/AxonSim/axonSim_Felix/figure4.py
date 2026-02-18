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




#Reading Matlab Scaled Data/simulated data and running calculation
#To process previously simulated data, run this

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
#fig 1:plot ave value for diff coe wiht error bars
#fig 1: show axon, 3D rendering and then showhow particle travels, both straight and beaded


def getNMRsignal(dxdist,dx,dTime):

#xarray, xbins, _ = plt.hist(d0, 200)
#plt.show()
#dx = (xbins[0:-1]+xbins[1:])/2

    #gg = np.linspace(0,2,100)  # 2*pi*gamma*g*delta, unit 1/um,
    gg = np.linspace(0,0.800,100)*2*np.pi*42.57e6*0.01/1e6
    bv = gg**2*(dTime-0.01/3)
    m0 = sum(dxdist)

    s = np.zeros_like(gg)
    for i in range(dx.shape[0]):
        s += dxdist[i]*np.cos(dx[i]*gg)
    s = s/m0
#s = np.sum(np.cos(np.outer(gg,dxlist)),1)
    # bv from s/um2 to s/mm2
    plt.plot(bv*1e6,s,'-')
    return bv*1e6, s, gg


#getNMRsignal(dxlistbig[0],dx,dTimelist[0])
bv1, d1,_ = getNMRsignal(dxdist[1],dx,dTimelist[1])
bv2, d2, _= getNMRsignal(dxdist[2],dx,dTimelist[2])
bv3, d3, _= getNMRsignal(dxdist[3],dx,dTimelist[3])
bv = np.logspace(0,5,100)
DC=.3e-3    # initial slope 0.5 for DC0=2000, with ldelta=10ms.
            # 0.3 for DC0=1000
            # 0.18 for DC0=500

plt.semilogy(bv,np.exp(-DC*bv),'r--')
plt.xlim(100,6e4)
plt.ylim(0.01 ,1.05)
plt.legend([dTimelist[1],dTimelist[2],dTimelist[3]])
plt.xlabel('b value, s/mm2')
plt.show
#%%

#%% 
set_num_threads(12)
dTimelist= np.array([0.01, 0.02, 0.05])


bvlist = np.zeros((dTimelist.shape[0],100))
slist  = np.zeros((dTimelist.shape[0],100))

i=0
for dTime in dTimelist:
    print(dTime)
    dx1, dy1, dz1,d0,d1,d2 = axon.bigData(2000, dTime,image,xyzres,ddir,pore)

    bvlist[i],slist[i] = axon.NMRsignal(d0,dTime)

    i+=1
#%
plt.semilogx(bvlist[0]*1e6,slist[0],'b')
plt.semilogx(bvlist[1]*1e6,slist[1],'m')
plt.semilogx(bvlist[2]*1e6,slist[2],'r')
#plt.semilogx(bvlist[3]*1e6,slist[3],'r')

plt.xlim((10,1e5))
plt.ylim((-0.1,1.1))
plt.xlabel('bvalue, s/mm2')
plt.ylabel('Simulated NMR signal')
plt.title(axonfilename+', dTime:'+str(dTimelist))
plt.show()

#
