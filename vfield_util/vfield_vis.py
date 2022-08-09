## vector field visualisation
## By Shizuo KAJI

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.axes_grid1
from pynhhd import nHHD
from scipy import interpolate,sparse
import matplotlib.colors as colors
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix

# plot function and contour
def plot_contour(f, X=None, Y=None, ax=None, levels=3, title=None, norm=None):
    px = np.arange(0,f.shape[1]) if X is None else X
    py = np.arange(0,f.shape[0]) if Y is None else Y
    if ax is None:
        fig, ax = plt.subplots(1,1)
    im = ax.imshow(f[::-1,:], cmap="coolwarm", norm=norm)
    if title is not None:
        ax.set_title(title)
    CS = ax.contour(px,py, f[::-1,:], levels=levels,colors='k')
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    cax = plt.colorbar(im, cax=cax)

# draw streamlines on a grid
def plot_slines(vx,vy, X=None, Y=None, ax=None, valrange=None, title=None, quiver=False, norm=None, density=(0.5,1), cmap="coolwarm"):
    mgn = np.sqrt(vx*vx+vy*vy)
    px = np.arange(0,vx.shape[1]) if X is None else X
    py = np.arange(0,vx.shape[0]) if Y is None else Y
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if quiver:
        n = 20
        scale = 350
        strm = ax.quiver(px[::n],py[::n],vx[::n],vx[::n],mgn,pivot='tail',cmap=cmap,scale=scale,scale_units='width',width=0.005)
    else:
        strm = ax.streamplot(px, py, vx,vy, color=mgn, linewidth=2, cmap=cmap, norm=norm,density=[0.5,1])
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"min {mgn.min():.4e}, max {mgn.max():.4e}")
    ax.set(xlim=(px.min(),px.max()),ylim=(py.min(),py.max()))
    #ax.set_aspect('equal')
    ax.set_aspect(1)
    ax.axis('off')
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    #cax.clim(vmin=mgn.min(),vmax=mgn.max)
    cax = plt.colorbar(strm.lines, cax=cax)

