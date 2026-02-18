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
from scipy.io import savemat

#Reading Matlab Scaled Data/simulated data and running calculation
#To process previously simulated data, run this

#Fast
import scipy.io
ldelta=[0,0.01];
DC0=600

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


#%%
#Created unscaled data
axonfile = '../../big_axon_matrix/axon_geo_all.mat'
matfile = mat73.loadmat(axonfile)
mat_all = matfile.get('axon_all')
for axons in [4, 909, 261, 249, 7, 24]:#[4, 909, 261, 249, 7, 24]
    mat = mat_all[axons]     
    pore = int32(np.array(mat.get('pore_voxels')))-1
    for n in nparticles:
        for t in range(4):
            #mat = mat73.loadmat(imagefilename_mat)
            #print('use mat73')
            volfactor = pore.shape[0]/n
            mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/ScaledData/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
            dx = mat['dx'].transpose()
            dxdist = mat['dxdist']/volfactor
            mdic = {"axon_name": "Axon " + str(axons), "DTime": dTimelist,"dxdist":dxdist,"ldelta":ldelta,"DC0":DC0,"dx":dx}
            savemat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/data/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat", mdic)
            
#fig 2: include units for x and y bar
#Normalize to one, label prob density
            
#%%
#Reading Matlab Raw Data/simulated data and running calculation
#Fast
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
            mat = scipy.io.loadmat("/Users/ysong/Desktop/Felix_2023/MGH-diff-time-corr/RawData/axon" + str(axons) + "/dxdist_trial" + str(t) + "_pnum" + str(n) + ".mat")
            dx = mat['dx']
            dxdist = mat['dxdist']
            dTimelist= np.array([0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,0.06,0.07,0.08,0.09,0.1])
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
#Figure 1a, looking at diffusion coefficient
axonNum = [4, 909, 261, 249, 7, 24]
for axons in range(6):#[4, 909, 261, 249, 7, 24]
    diffAve = np.zeros(len(dTimelist))
    varAve = np.zeros(len(dTimelist))
    for t in range(4):
        varAve += (dataBig[axons][3][t]-aveBig[axons])**2
        diffAve += dataBig[axons][3][t]
    for a in range(15): varAve[a] = math.sqrt(varAve[a]/4)
    plt.plot(dTimelist, diffAve/4, "-o", markersize = 3)
    plt.errorbar(dTimelist, diffAve/4, yerr=varAve)
    plt.title("Axon:" + str(axonNum[axons]) + " Diffusion coefficient as time to diffuse increases")
    plt.xlabel("dTime")
    plt.ylabel("Diffusion Coefficient")
    plt.show()
    



#%%
#Calculaing ave variance of axon at different particle counts
axonIndex = 0
for num in [0,1,2,3]:
    vv=0
    for runs in [0,1,2,3]:
        vv += np.sum((dataBig[axonIndex][num][runs]-aveBig[axonIndex])**2)
    vv = math.sqrt(vv/14/4)
    print(nparticles[num], ":", vv)




#%%

# Plot xmean and x-squared
plt.plot(dTimelist, x_mean, '-o')
plt.title("xmean graph")
plt.xlabel("diffusion time - s")
plt.ylabel("x mean - um")
plt.show()

plt.plot(dTimelist, xsq_mean, '-o')
plt.title("x squared mean graph")
plt.xlabel("diffusion time - s")
plt.ylabel("x square mean- um^2")
plt.show()
    




