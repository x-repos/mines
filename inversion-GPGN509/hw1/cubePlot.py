import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

plt.rc('text', usetex=True)
font = {'color':'black','size':16}
plt.rcParams.update({'font.size': 16})
    
def plot1D(x,fx, col='k',lx="", title=""):
    plt.plot(x,fx,col, lw=3)
    plt.xlabel(lx,fontdict=font)
    plt.xlim(min(x),max(x))
    plt.title(title)
    plt.grid()
    
def plot2D(x,y,fxy, 
           lx="",ly="",
           title="",
           pclip=0.95):
    
    X,Y = np.meshgrid(x,y, indexing="ij")
    
    cmin = np.min(fxy)
    cmax = np.max(fxy) * pclip
    
    plt.contourf(X,Y,fxy, np.linspace(cmin,cmax,100), extend='max', cmap='Blues')
    plt.colorbar()
    
    plt.xlabel(lx,fontdict=font)
    plt.ylabel(ly,fontdict=font)
    plt.title(title)

def get_the_slice(x,y,z,c):
    return go.Surface(x = x,
                      y = y,
                      z = z,
                      surfacecolor = c,
                      coloraxis = 'coloraxis',
                      opacity = 0.25)

def plot3D(x,y,z,fxyz, 
           jx=None,jy=None,jz=None,
           fx=0,   fy=0,   fz=0,
           lx="",  ly="",  lz="",
           title="",
           pclip=0.75):
    
    if jx==None and jy==None and jz==None:
        jx = int(len(x)/10)
        
    XX,YY,ZZ = np.meshgrid(x,y,z, indexing="ij")
    
    slices = []
      
    # X slices
    if jx != None:
        Y,Z = np.meshgrid(y,z, indexing="ij")
        for ix in range(fx,len(x),jx):
            X =   XX[ix,:,:]
            C = fxyz[ix,:,:]
            slices.append(get_the_slice(X,Y,Z,C))
        
    # Y slices
    if jy != None:
        X,Z = np.meshgrid(x,z, indexing="ij")
        for iy in range(fy,len(y),jy):            
            Y =   YY[:,iy,:]
            C = fxyz[:,iy,:]
            slices.append(get_the_slice(X,Y,Z,C))
    
    # Z slices
    if jz != None:
        X,Y = np.meshgrid(x,y, indexing="ij")
        for iz in range(fz,len(z),jz):            
            Z =   ZZ[:,:,iz]
            C = fxyz[:,:,iz]
            slices.append(get_the_slice(X,Y,Z,C))
        
    # begin plot #####################################

    fig = go.Figure( data=slices )
    
    par = dict()
 
    par['width']  = 800
    par['height'] = 800
    
    par['title_text'] = title
    par['scene'] = dict( xaxis_title = lx, 
                         yaxis_title = ly, 
                         zaxis_title = lz)
    
    par['scene_xaxis_range'] = [min(x),max(x)]
    par['scene_yaxis_range'] = [min(y),max(y)]
    par['scene_zaxis_range'] = [min(z),max(z)]
    
    par['coloraxis'] = dict(colorscale = 'Blues',
                            colorbar_thickness = 10,
                            colorbar_len = 0.5,
                            cmin = np.min(fxyz),
                            cmax = np.max(fxyz) * pclip)
    
    return fig.update_layout(par)