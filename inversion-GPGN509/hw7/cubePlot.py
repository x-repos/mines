import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo


plt.rc('text', usetex=False)
# plt.rcParams['text.usetex'] = False

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
           pclip=0.75,
           aspect_ratio=None):
    
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
    
    par['coloraxis'] = dict(colorscale = 'viridis',
                            colorbar_thickness = 10,
                            colorbar_len = 0.5,
                            cmin = np.min(fxyz),
                            cmax = np.max(fxyz) * pclip)
    
    # Set aspect ratio if provided
    if aspect_ratio:
        par['scene']['aspectmode'] = 'manual'
        par['scene']['aspectratio'] = dict(
            x=aspect_ratio[0], 
            y=aspect_ratio[1], 
            z=aspect_ratio[2]
        )
    else:
        par['scene']['aspectmode'] = 'auto'
    
    
    return fig.update_layout(par)

def plot_3d_line(x, y, z, title='3D Line Plot', x_label='X Axis', y_label='Y Axis', z_label='Z Axis', aspect_ratio=None):
    """
    Create a 3D line plot.

    Parameters:
        x (array-like): Array of x values.
        y (array-like): Array of y values.
        z (array-like): Array of z values.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        z_label (str): Label for the z-axis.
        aspect_ratio (dict): Dictionary to set aspect ratio with keys 'x', 'y', 'z'.
    """
    
    # Create a 3D line trace
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines',                     # Use 'lines' for line plot
        line=dict(color='blue', width=2)  # Customize the line color and width
    )

    # Define layout
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label),
            zaxis=dict(title=z_label),
        )
    )
    # Set aspect ratio if provided
    if aspect_ratio:
        layout['scene']['aspectmode'] = 'manual'
        layout['scene']['aspectratio'] = dict(
            x=aspect_ratio[0], 
            y=aspect_ratio[1], 
            z=aspect_ratio[2]
        )
    else:
        layout['scene']['aspectmode'] = 'auto'

    # Create a figure
    fig = go.Figure(data=[trace], layout=layout)


    # Create a figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the plot
    pyo.iplot(fig)  # Use py


    import numpy as np
import plotly.graph_objects as go


def plot3D1(x, y, z, fxyz, 
           jx=None, jy=None, jz=None,
           fx=0, fy=0, fz=0,
           lx="", ly="", lz="",
           title="",
           pclip=0.75,
           aspect_ratio=None,
           line_x=None, line_y=None, line_z=None):  # Added line parameters
    
    if jx is None and jy is None and jz is None:
        jx = int(len(x) / 10)
        
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
    
    slices = []
      
    # X slices
    if jx is not None:
        Y, Z = np.meshgrid(y, z, indexing="ij")
        for ix in range(fx, len(x), jx):
            X = XX[ix, :, :]
            C = fxyz[ix, :, :]
            slices.append(get_the_slice(X, Y, Z, C))
        
    # Y slices
    if jy is not None:
        X, Z = np.meshgrid(x, z, indexing="ij")
        for iy in range(fy, len(y), jy):            
            Y = YY[:, iy, :]
            C = fxyz[:, iy, :]
            slices.append(get_the_slice(X, Y, Z, C))
    
    # Z slices
    if jz is not None:
        X, Y = np.meshgrid(x, y, indexing="ij")
        for iz in range(fz, len(z), jz):            
            Z = ZZ[:, :, iz]
            C = fxyz[:, :, iz]
            slices.append(get_the_slice(X, Y, Z, C))
        
    # Begin plot #####################################
    fig = go.Figure(data=slices)
    
    par = dict()
 
    par['width'] = 800
    par['height'] = 800
    
    par['title_text'] = title
    par['scene'] = dict(xaxis_title=lx, 
                        yaxis_title=ly, 
                        zaxis_title=lz)
    
    par['scene_xaxis_range'] = [min(x), max(x)]
    par['scene_yaxis_range'] = [min(y), max(y)]
    par['scene_zaxis_range'] = [min(z), max(z)]
    
    par['coloraxis'] = dict(colorscale='Oranges',
                            colorbar_thickness=10,
                            colorbar_len=0.5,
                            cmin=np.min(fxyz),
                            cmax=np.max(fxyz) * pclip)
    
    # Set aspect ratio if provided
    if aspect_ratio:
        par['scene']['aspectmode'] = 'manual'
        par['scene']['aspectratio'] = dict(
            x=aspect_ratio[0], 
            y=aspect_ratio[1], 
            z=aspect_ratio[2]
        )
    else:
        par['scene']['aspectmode'] = 'auto'

    # Plot the line if coordinates are provided
    if line_x is not None and line_y is not None and line_z is not None:
        fig.add_trace(go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode='lines',
            line=dict(color='red', width=6),  # Use a different color for visibility
            name='Line'
        ))
    
    return fig.update_layout(par)
