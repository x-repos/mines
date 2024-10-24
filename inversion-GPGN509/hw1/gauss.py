import numpy as np
from numpy.linalg import inv,det
from scipy.stats import multivariate_normal

# 1D Gaussian
def gauss1D(x,xCEN,xSTD):
        
    GAU = np.exp( -0.5*((x - xCEN)/xSTD)**2 )
    GAU /= np.sqrt( 2*np.pi * xSTD**2 )
    
    return GAU

# 2D Gaussian (independent variables)
def gauss2D(x,xCEN,xSTD, 
            y,yCEN,ySTD, fast=True):
    
    xGAU = gauss1D(x,xCEN,xSTD)
    yGAU = gauss1D(y,yCEN,ySTD)
    
    if fast:
        GAU = np.outer(xGAU,yGAU)
        
    else:
        GAU = np.zeros( (len(x),len(y)) )
        for iy in range(len(y)):
            for ix in range(len(x)):
                GAU[ix,iy] = xGAU[ix] * yGAU[iy]

    return GAU

# 2D Gaussian (correlated variables)
def gcorr2D(x,xCEN,xSTD, 
            y,yCEN,ySTD, 
            rxy=0,
            fast=True):
    
    C = np.array([[xSTD**2, rxy*xSTD*ySTD],
                  [rxy*xSTD*ySTD, ySTD**2]])
    
    if fast:
        X, Y = np.meshgrid(x,y, indexing="ij")
        pos = np.stack((X, Y), axis=-1)
        GAU = multivariate_normal.pdf(pos, [xCEN, yCEN], C)
    
    else:
        IC = inv(C)

        GAU = np.zeros( (len(x),len(y)) )
        for iy in range(len(y)):
            for ix in range(len(x)):
                l = np.array([x[ix]-xCEN, y[iy]-yCEN])
                GAU[ix,iy] = np.exp(-0.5 * l @ (IC @ l.T) )

        GAU /= np.sqrt( np.power(2*np.pi,2) * det(C) )

    return GAU

# 3D Gaussian (independent variables)
def gauss3D(x,xCEN,xSTD, 
            y,yCEN,ySTD, 
            z,zCEN,zSTD, fast=True):
    
    xGAU = gauss1D(x,xCEN,xSTD)
    yGAU = gauss1D(y,yCEN,ySTD)
    zGAU = gauss1D(z,zCEN,zSTD)
    
    if fast:
        GAU = np.outer(xGAU, np.outer(yGAU,zGAU))
        GAU = np.reshape(GAU, (len(x),len(y),len(z)))
    
    else:
        GAU = np.zeros( (len(x),len(y),len(z)) )
        for ix in range(len(x)):
            for iy in range(len(y)):
                for iz in range(len(z)):
                    GAU[ix,iy,iz] = xGAU[ix] * yGAU[iy] * zGAU[iz]
    
    return GAU

# 3D Gaussian (correlated variables)
def gcorr3D(x,xCEN,xSTD, 
            y,yCEN,ySTD, 
            z,zCEN,zSTD, 
            rxy=0,rxz=0,ryz=0,
            fast=True):
    
    C = np.array([[xSTD**2, rxy*xSTD*ySTD, rxz*xSTD*zSTD],
                  [rxy*xSTD*ySTD, ySTD**2, ryz*ySTD*zSTD],
                  [rxz*xSTD*zSTD, ryz*ySTD*zSTD, zSTD**2]])
    
    if fast:
        X, Y, Z = np.meshgrid(x,y,z, indexing="ij")
        pos = np.stack((X, Y, Z), axis=-1)
        GAU = multivariate_normal.pdf(pos, [xCEN, yCEN, zCEN], C)
    
    else:
        IC = inv(C)

        GAU = np.zeros( (len(x),len(y),len(z)) )
        for iz in range(len(z)):
            for iy in range(len(y)):
                for ix in range(len(x)):
                    l = np.array([x[ix]-xCEN, y[iy]-yCEN, z[iz]-zCEN])
                    GAU[ix,iy,iz] = np.exp(-0.5 * l @ (IC @ l.T) )

        GAU /= np.sqrt( np.power(2*np.pi,3) * det(C) )
            
    return GAU