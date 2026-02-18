import scipy.io

# Felix Song, May -, 2022
# Diffusion simulation on axon images 3d
# Aug 13, 2022. move to python folder. Only the core code.

# basic functions for simulation

# Sept 3, add little-delta in the displacement calculation. YS
# Nov 18, add DC into the bigData input
# add a demo function to showcase the simulation

__version__='0.9.0'
print('AxonSim3D, 090new, ver:', __version__)

#Using original data
from numba import jit, prange, set_num_threads
import numpy as np
import time
import math
from numba import int32, uint32, uint8, float32, uint16   # import the types
from numba.experimental import jitclass
import matplotlib.pyplot as plt


# the data record for a walker class
spec = [
    ('n', int32),
    ('istep', int32),
    ('x', int32),                   # unit of voxel index
    ('y', int32),       
    ('z', int32),
    ('dt',float32),
    ('DC',float32),
    ('dstep',float32),
    ('xtrace', int32[:]),   #unit of voxel index
    ('ytrace', int32[:]),
    ('ztrace', int32[:]),
    ('xres', float32),
    ('yres', float32),
    ('zres', float32),
    ('axon', uint8[:,:,:]),   
    ('ddir', float32[:,:]),
    ('pore', int32[:,:])
]

# definition of the walker class to perform simulations
@jitclass(spec) #,parallel=True)
class walker(object):
    def __init__(self, n=100,  axon=uint8[:,:,:], xyzres=[0.032,0.032,0.03],ddir=float32[:,:], pore=uint32[:,:]):
        self.n = n
        self.axon = axon
        self.istep = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.dt = 1e-6
        self.DC = 2000
        self.dstep = np.sqrt(2*self.DC*self.dt)
        self.xtrace = np.zeros(n, dtype=np.int32)
        self.ytrace = np.zeros(n, dtype=np.int32)
        self.ztrace = np.zeros(n, dtype=np.int32)
        self.xres = 0.032 #from index amount to microns, from qiyuan Tian, QTIAN@mgh.harvard.edu
        self.yres = 0.032
        self.zres = 0.03
        # need to get the resolution from the image 
        self.xres = xyzres[0]
        self.yres = xyzres[1]
        self.yres = xyzres[2]
        # ddir contains 3 directions to calculate displacement
        self.ddir = np.zeros((3,3), dtype=np.float32)
        self.ddir[0] = ddir[0] #np.array(ddir,dtype=np.float32)
        self.ddir[1] = ddir[1] #np.array(ddir,dtype=np.float32)
        self.ddir[2] = ddir[2] #np.array(ddir,dtype=np.float32)
        #print('self.ddif:\n',self.ddir)
        #print('input ddif:\n',ddir)
        self.pore = pore    # list of the coordinates of the pore points
        
        # avoid reading axon matrix
#        for i in range(len(pore)):
#            x,y,z = pore[i]
#            self.axon[x,y,z]=1
        
        
    @property
    def size(self):
        return self.n
    
    # choose a random position in the axon to start the simulation
    def startPoint(self):
       num = int(np.random.random()*self.pore.shape[0])
       return self.pore[num]

    def increment(self, nstep=1):
        for i in range(nstep):
            if self.istep < self.n :
                self.x += int32(np.round(self.dstep * np.random.randn()/self.xres))
                self.y += int32(np.round(self.dstep * np.random.randn()/self.yres))
                self.z += int32(np.round(self.dstep * np.random.randn()/self.zres))

                self.xtrace[self.istep] = self.x
                self.ytrace[self.istep] = self.y
                self.ztrace[self.istep] = self.z
                self.istep +=1

    # diffuse the walker for an amount of time (dtime) and return the trajectory or the final displacement
    # along XYZ and the a few given directions
    # add little-delta: ldelta, default to zero
    def displacement(self,dtime,ldelta=0):
        nstep = int32((dtime+ldelta)/self.dt)
        self.dstep = np.sqrt(2*self.DC*self.dt)
        
        self.xtrace = np.zeros(nstep, dtype=np.int32)
        self.ytrace = np.zeros(nstep, dtype=np.int32)
        self.ztrace = np.zeros(nstep, dtype=np.int32)
        
        x0 = self.x
        y0 = self.y
        z0 = self.z
        i=0
        self.xtrace[i] = self.x
        self.ytrace[i] = self.y
        self.ztrace[i] = self.z
        
        dstepx = self.dstep/self.xres
        dstepy = self.dstep/self.yres
        dstepz = self.dstep/self.zres
        
        for i in range(1,nstep):
            x = self.x
            y = self.y
            z = self.z
            dx = int32(np.round(dstepx * np.random.randn()))
            dy = int32(np.round(dstepy * np.random.randn()))
            dz = int32(np.round(dstepz * np.random.randn()))
            while self.inAxon(x+dx, y+dy, z+dz) is False:
                dx = int32(np.round(dstepx * np.random.randn()))
                dy = int32(np.round(dstepy * np.random.randn()))
                dz = int32(np.round(dstepz * np.random.randn()))
                
            self.x += dx
            self.y += dy
            self.z += dz
            
            self.xtrace[i] = self.x
            self.ytrace[i] = self.y
            self.ztrace[i] = self.z
        
        if ldelta==0:
            dd = np.array([(self.x - x0)*self.xres, (self.y - y0)*self.yres, (self.z - z0)*self.zres],dtype=np.float32)
        else:
            ndelta = int32(ldelta/self.dt)
            nbigdelta = int32(dtime/self.dt)
            dx0=np.mean(self.xtrace[0:ndelta]) - np.mean(self.xtrace[nbigdelta:])
            dy0=np.mean(self.ytrace[0:ndelta]) - np.mean(self.ytrace[nbigdelta:])
            dz0=np.mean(self.ztrace[0:ndelta]) - np.mean(self.ztrace[nbigdelta:])
            dd = np.array([(dx0)*self.xres, (dy0)*self.yres, (dz0)*self.zres],dtype=np.float32)
        
        #ddd = np.dot(dd,np.transpose(self.ddir))
        dd0 = np.dot(dd,self.ddir[0])
        dd1 = np.dot(dd,self.ddir[1])
        dd2 = np.dot(dd,self.ddir[2])
        
        #return self.x - x0, self.y - y0, self.z - z0, (self.x - x0)*self.xres, (self.y - y0)*self.yres, (self.z - z0)*self.zres
        return (self.x - x0)*self.xres, (self.y - y0)*self.yres, (self.z - z0)*self.zres,dd0,dd1,dd2
    
    def displacement2(self,dtime : float32):
        nstep = math.floor(dtime/self.dt)
        self.xtrace = np.zeros(nstep, dtype=np.int32)
        self.ytrace = np.zeros(nstep, dtype=np.int32)
        self.ztrace = np.zeros(nstep, dtype=np.int32)
        
        x1 = self.x
        y1 = self.y
        z1 = self.z
        i=0
        self.xtrace[i] = self.x
        self.ytrace[i] = self.y
        self.ztrace[i] = self.z
        for i in range(nstep+1,2*nstep):
            x = self.x
            y = self.y
            z = self.z
            dx = int32(np.round(self.dstep * np.random.randn()/self.xres))
            dy = int32(np.round(self.dstep * np.random.randn()/self.yres))
            dz = int32(np.round(self.dstep * np.random.randn()/self.zres))
            while self.inAxon(x+dx, y+dy, z+dz) is False:
                dx = int32(np.round(self.dstep * np.random.randn()/self.xres))
                dy = int32(np.round(self.dstep * np.random.randn()/self.yres))
                dz = int32(np.round(self.dstep * np.random.randn()/self.zres))
                
            self.x += dx
            self.y += dy
            self.z += dz
            
            self.xtrace[i] = self.x
            self.ytrace[i] = self.y
            self.ztrace[i] = self.z

        #return self.x - x1, self.y - y1, self.z - z1
        return (self.x - x1)*self.xres, (self.y - y1)*self.yres, (self.z - z1)*self.zres
    
    def inAxon(self,x,y,z)->bool:
        if (x >= 0) and (x < self.axon.shape[0]) and (y >= 0) and (y < self.axon.shape[1]) and (z >= 0) and (z < self.axon.shape[2]):
            return self.axon[x,y,z] == 1
        else:
            return False
           
            
    def inAxon2(self,x,y,z) -> bool:
        return self.inAxon(self.x, self.y, self.z)

    def get1stpt(self):
        x, y, z = self.startPoint()
        while 1:
            x0 = x + int32(np.round(self.dstep * np.random.randn()/self.xres))
            y0 = y + int32(np.round(self.dstep * np.random.randn()/self.yres))
            z0 = z + int32(np.round(self.dstep * np.random.randn()/self.zres))
            if self.inAxon(x0,y0,z0):
                self.x = x0
                self.y = y0
                self.z = z0
                break
                
    def get1stpt2(self):
        x, y, z = self.startPoint()
        while 1:
            x1 = x + int32(np.round(self.dstep * np.random.randn()/self.xres))
            y1 = y + int32(np.round(self.dstep * np.random.randn()/self.yres))
            z1 = z + int32(np.round(self.dstep * np.random.randn()/self.zres))
            if self.inAxon(x1,y1,z1):
                self.x = x1
                self.y = y1
                self.z = z1
                break

    def show(self):
        N = 50000
        xlist = np.zeros(N)
        ylist = np.zeros(N)
        zlist = np.zeros(N)
        x, y, z = self.startPoint()
        x0 = x
        y0 = y
        z0 = z
        for i in range(N):
            x1 = x0 + int32(np.round(self.dstep * np.random.randn()/self.xres))
            y1 = y0 + int32(np.round(self.dstep * np.random.randn()/self.yres))
            z1 = z0 + int32(np.round(self.dstep * np.random.randn()/self.zres))
            while not(self.inAxon(x1,y1,z1)):
                
                x1 = x0 + int32(np.round(self.dstep * np.random.randn()/self.xres))
                y1 = y0 + int32(np.round(self.dstep * np.random.randn()/self.yres))
                z1 = z0 + int32(np.round(self.dstep * np.random.randn()/self.zres))
            
            xlist[i] = x1
            ylist[i] = y1
            zlist[i] = z1
            x0 = x1
            y0 = y1
            z0 = z1
        return xlist, ylist, zlist


    def show1(self):
        N = 50000
        xlist = np.zeros(N)
        ylist = np.zeros(N)
        zlist = np.zeros(N)

        for i in range(N):
            x1, y1, z1 = self.startPoint()
            
            xlist[i] = x1
            ylist[i] = y1
            zlist[i] = z1
        return xlist, ylist, zlist

                
    def dsim3d(self, dtime:int, nparticles:uint8):
        dxlist = np.zeros(nparticles)
        dylist = np.zeros(nparticles)
        dzlist = np.zeros(nparticles)
        xlist = np.zeros(nparticles)
        ylist = np.zeros(nparticles)
        zlist = np.zeros(nparticles) 
        
        for i in range(nparticles):
            self.get1stpt()
            dxlist[i], dylist[i], dzlist[i] = self.displacement(dtime)
            xlist[i] = self.x
            ylist[i] = self.y
            zlist[i] = self.z

            
        dxlist2 = np.zeros(nparticles)
        dylist2 = np.zeros(nparticles)
        dzlist2 = np.zeros(nparticles)
        xlist2 = np.zeros(nparticles)
        ylist2 = np.zeros(nparticles)
        zlist2 = np.zeros(nparticles) 
        
        for i in range(nparticles):
            self.get1stpt2()
            dxlist2[i], dylist2[i], dzlist2[i] = self.displacement2(dtime)
            xlist2[i] = self.x
            ylist2[i] = self.y
            zlist2[i] = self.z

            
        return dxlist, dylist, dzlist, dxlist2, dylist2, dzlist2

        
    @staticmethod
    def add(x, y, z):
        return x + y + z
# END OF BASIC DEFINITIONS of the simulation


# a demo function
@jit(nopython=True, parallel=True)
def demo():
    axon = np.zeros((1000,1000,1000),'uint8')
    axon[:,450:550,450:550]=1
    xyzres = np.array([30,30,30])/1000
    ddir =np.array([[1,0,0],[0,1,0],[0,0,1]])
    pore = np.array(np.argwhere(axon==1),dtype=np.int32)
    #pore = np.array(np.nonzero(axon==1))
    
    print (ddir)
    print(axon.shape)
    #print(pore.shape)
    
    ww = walker(100,axon,xyzres,ddir,(pore))
    
    x,y,z =ww.show1()

# # 3D plot
    xx, yy, zz = np.broadcast_arrays(x, y, z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
                y.ravel(),
                z.ravel(),
                c=zz)
    plt.xlim(0,1000)
    plt.ylim(0,1000)
    #plt.set_zlim(0,1000)
    ax.set_zlim(0,1000)
    plt.show()
    
    nparticles = 10000
    dxlist = np.zeros(nparticles)
    dylist = np.zeros(nparticles)
    dzlist = np.zeros(nparticles)
    
    d0 = np.zeros(nparticles)
    d1 = np.zeros(nparticles)
    d2 = np.zeros(nparticles)
    
    dtime = 0.01
    ldelta=0
    
    for i in prange(nparticles):      
        walker1 = walker(100,axon,xyzres,ddir,pore)
        walker1.DC = 1000
        walker1.get1stpt()
        #print('run displacement ...',i)
        dxlist[i], dylist[i], dzlist[i],d0[i],d1[i],d2[i] = walker1.displacement(dtime,ldelta)

    plt.hist(d0, 100)
    plt.show()
    return d0

# more code to perform simulation
#analyzing more than one walker and adding up data to accumulate statistics
# Fitting function: Gaussian form

from scipy.optimize import curve_fit
def fitfunc(dx, a, sigma):
    return a * np.exp(-dx**2/2/sigma**2)

# do the simulation for one diffusion time
# try parallel this:
@jit(nopython=True, parallel=True)
def bigData(nparticles,dtime,image,xyzres,ddir,pore,ldelta=0,DC0=2000):
    
    dxlist = np.zeros(nparticles)
    dylist = np.zeros(nparticles)
    dzlist = np.zeros(nparticles)
    
    d0 = np.zeros(nparticles)
    d1 = np.zeros(nparticles)
    d2 = np.zeros(nparticles)
    
    
    for i in prange(nparticles):      
        walker1 = walker(100,image,xyzres,ddir,pore)
        walker1.DC = DC0
        walker1.get1stpt()
        #print('run displacement ...',i)
        dxlist[i], dylist[i], dzlist[i],d0[i],d1[i],d2[i] = walker1.displacement(dtime,ldelta)
    
    
    return dxlist, dylist, dzlist,d0,d1,d2


def NMRsignal(dxlist,dTime):
    
    #xarray, xbins, _ = plt.hist(d0, 200)
    #plt.show()
    #dx = (xbins[0:-1]+xbins[1:])/2
    
    gg = np.linspace(0,2,100)  # 2*pi*gamma*g*delta, unit 1/um,
    bv = gg**2*dTime
    m0 = dxlist.shape[0]
    
    #s = np.zeros_like(gg)
    #for i in range(dx.shape[0]):
    #    s += xarray[i]*np.cos(dx[i]*gg)
        
    s = np.sum(np.cos(np.outer(gg,dxlist)),1)
    
    plt.plot(bv*1e6,s/m0,'.')
    plt.xlabel('bvalue, s/mm2')
    plt.title('Sim NMR: Diffusion time:'+str(dTime)+' s')
    plt.show()
    return bv, s/m0

#plot of D and delta
#timeArray = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

# Do simulation for a list of diffusion time, and fit for diffusion coefficient as a function of the diffusion time
def pltdeltaD(timeList, nparticles,imagefilename_mat,ldelta=0):
    # read the matlab file
    mat = scipy.io.loadmat(imagefilename_mat) 

    image = np.array(mat.get('BB'))
    print('Image size:\n', image.shape)
    
    xyzres = np.array(mat.get('res_um'))[0]
    print('image res:\n', xyzres)
    
    # read the struct pp.vec
    ddir = np.array(mat.get('pp'))['vec'][0][0].transpose()
    print('ddir:\n',ddir)    
    
    pore = np.array(mat.get('pore_voxels'))-1	# -1 to convert matlab index to python index
    print('pore:\n',pore.shape())
    
    #mybins = np.linspace(-100 ,100,201)
    DxList = np.zeros(len(timeList))
    DyList = np.zeros(len(timeList))
    DzList = np.zeros(len(timeList))
    
    D0List = np.zeros(len(timeList))
    D1List = np.zeros(len(timeList))
    D2List = np.zeros(len(timeList))
    
    a = 0
    for dtime in timeList:
        print ('diffusion time:',dtime)
        
        t0=time.perf_counter()
        dx1, dy1, dz1, d0,d1,d2 = bigData(nparticles, dtime,image,xyzres,ddir,pore,ldelta)
        
        # do the histogram for 50 bins. Let hist to determine where the bins are
        # xarray, xbins, _ = plt.hist(dx1, 50)
        # yarray, ybins, _ = plt.hist(dy1, 50)
        # zarray, zbins, _ = plt.hist(dz1, 50)
        # poptx, pcovx = curve_fit(fitfunc, (xbins[0:-1]+xbins[1:])/2, xarray)
        # popty, pcovy = curve_fit(fitfunc, (ybins[0:-1]+ybins[1:])/2, yarray)
        # poptz, pcovz = curve_fit(fitfunc, (zbins[0:-1]+zbins[1:])/2, zarray)
        
        # DxList[a] = (poptx[1])**2/2/dtime
        # DyList[a] = (popty[1])**2/2/dtime
        # DzList[a] = (poptz[1])**2/2/dtime
        
        # calculate D via <r2>/2dtime
        # also Gaussian fit
        
        xarray, xbins, _ = plt.hist(d0, 50)
        poptx, pcovx = curve_fit(fitfunc, (xbins[0:-1]+xbins[1:])/2, xarray)
        D0List[a] = (poptx[1])**2/2/dtime
        dx = np.array((xbins[0:-1]+xbins[1:])/2)
        DxList[a] = np.sum(dx**2*xarray)/2/dtime/np.sum(xarray)
        
        xarray, xbins, _ = plt.hist(d1, 50)
        poptx, pcovx = curve_fit(fitfunc, (xbins[0:-1]+xbins[1:])/2, xarray)
        D1List[a] = (poptx[1])**2/2/dtime
        dx = np.array((xbins[0:-1]+xbins[1:])/2)
        DyList[a] = np.sum(dx**2*xarray)/2/dtime/np.sum(xarray)
        
        xarray, xbins, _ = plt.hist(d2, 50)
        poptx, pcovx = curve_fit(fitfunc, (xbins[0:-1]+xbins[1:])/2, xarray)
        D2List[a] = (poptx[1])**2/2/dtime
        dx = np.array((xbins[0:-1]+xbins[1:])/2)
        DzList[a] = np.sum(dx**2*xarray)/2/dtime/np.sum(xarray)
                       
        print('Elapsed time(s):',time.perf_counter()-t0)
        a += 1
    return DxList, DyList, DzList, D0List, D1List, D2List


#%%

# timeList = [0.001,0.005]
# nparticles = 1000
# imagefilename = 'axon_129.mat'
# matTest = scipy.io.loadmat(imagefilename) 
# 
# Dx, Dy, Dz, Dx0, Dy0, Dz0 = pltdeltaD(timeList, nparticles, imagefilename)
# 
# 
# #%% ---- 
# mat = scipy.io.loadmat(imagefilename_mat) 
# 
# image = np.array(mat.get('BB'))
# print('Image size:\n', image.shape)
# 
# xyzres = np.array(mat.get('res_um'))[0]
# print('image res:\n', xyzres)
# 
# # read the struct pp.vec
# ddir = np.array(mat.get('pp'))['vec'][0][0].transpose()
# print('ddir:\n',ddir)    
# 
# pore = np.array(mat.get('pore_voxels'))
# print('pore:\n',pore)
# 
# walker1 = walker(100,image,xyzres,ddir,pore)
# walker1.get1stpt()

#%% make a test code

if 0:
    image = np.zeros((1000,1000,1000),dtype=np.uint8)
    image[:,500:550,500:550] = 1
    
    xyzres = np.array([0.03,0.03,0.03])
    
    ddir = np.zeros((3,3))
    
    ddir[0]=np.array([1,0,0])
    ddir[1]=np.array([0,1,0])
    ddir[2]=np.array([0,0,1])
    
    pore = np.uint32(np.argwhere(image==1))
    
    
    walker1 = walker(100,image,xyzres,ddir,pore)
    # # 3D plot
    x,y,z = walker1.show1()
    #xx, yy, zz = np.broadcast_arrays(x, y, z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
                y.ravel(),
                z.ravel())
    plt.xlim((0,1000))
    plt.ylim((0,1000))
    ax.set_zlim((0,1000))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    #%
    
    #%
    nparticles = 1000
    dtime= 0.01
    ldelta=0.005
    dx,dy,dz,d0,d1,d2 = bigData(nparticles,dtime,image,xyzres,ddir,pore,ldelta=ldelta)
    #%
    plt.plot(d0,'r')
    plt.plot(dy,'b')
    plt.plot(dz,'g')
    
    plt.show()
