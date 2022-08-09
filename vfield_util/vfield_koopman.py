## Koopman operator for vector field
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


# infinitesimal Koopman generator
def koopman_L(vx,vy,dx=(1,1), constraint_idx=[], constraint_weight=1e+3):
  ny,nx = vx.shape
  ## grad_x * vx
  Gx = sparse.dok_matrix((nx*ny+len(constraint_idx),nx*ny))
  for i,idx in enumerate(constraint_idx):
    Gx[(nx)*(ny)+i,idx] = constraint_weight ## enforce the value of phi[0] (killing ambiguity by a scalar multiple)

  for y in range(ny):
    Gx[y*nx, y*nx+1] = vx[y,0]/dx[1] # coeff of f(1,y) in df(0,y)
    Gx[y*nx, y*nx] = -vx[y,0]/dx[1] # (0,y) -> (0,y)
    Gx[y*nx+nx-1, y*nx+nx-1] = vx[y,nx-1]/dx[1] # (nx-1,y) -> (nx-1,y)
    Gx[y*nx+nx-1, y*nx+nx-2] = -vx[y,nx-1]/dx[1] #(nx-2,y) -> (nx-1,y)
    for x in range(1,nx-1):
      Gx[y*nx+x, y*nx+x+1] = 0.5*vx[y,x]/dx[1]
      Gx[y*nx+x, y*nx+x-1] = -0.5*vx[y,x]/dx[1]

  ## grad_y * vy
  Gy = sparse.dok_matrix((nx*ny+len(constraint_idx),nx*ny))
  for x in range(nx):
    Gy[x, 1*nx+x] = vy[0,x]/dx[0]
    Gy[x, x] = -vy[0,x]/dx[0]
    Gy[(ny-1)*nx+x, (ny-1)*nx+x] = vy[ny-1,x]/dx[0]
    Gy[(ny-1)*nx+x, (ny-2)*nx+x] = -vy[ny-1,x]/dx[0]
    for y in range(1,ny-1):
      Gy[y*nx+x, (y+1)*nx+x] = 0.5*vy[y,x]/dx[0]
      Gy[y*nx+x, (y-1)*nx+x] = -0.5*vy[y,x]/dx[0]

  #return(Gx.todense(), Gy.todense())
  return(Gx.tocsr() + Gy.tocsr())
  

# infinitesimal Koopman generator with a specified 3x3 gradient kernel
def koopman_Lab(vx,vy,dx=(1,1),_a=10,_b=3, constraint_idx=[], constraint_weight=1e+3):
    # Scharr filter (a,b)=(10,3)
    # Sobel filter  (a,b)=(2,1)
  ny,nx = vx.shape
  a,b = _a,_b
  c = a+2*b
  a /= c
  b /= c
  a2 = a/(a+b)
  b2 = b/(a+b)
  ## grad_x * vx
  Gx = sparse.dok_matrix(((nx)*(ny)+len(constraint_idx),(nx)*(ny)))
  for i,idx in enumerate(constraint_idx):
    Gx[(nx)*(ny)+i,idx] = constraint_weight ## enforce the value of phi[0] (killing ambiguity by a scalar multiple)

  ## (y,x)=(0,0)
  Gx[0, 1] = 2*a2*vx[0,0]/dx[1]
  Gx[0, 0] = -2*a2*vx[0,0]/dx[1]
  Gx[0, nx+1] = 2*b2*vx[0,0]/dx[1]
  Gx[0, nx] = -2*b2*vx[0,0]/dx[1]
  ## (y,x)=(0,nx-1)
  Gx[nx-1, nx-1] = 2*a2*vx[0,nx-1]/dx[1]
  Gx[nx-1, nx-2] = -2*a2*vx[0,nx-1]/dx[1]
  Gx[nx-1, 2*nx-1] = 2*b2*vx[0,nx-1]/dx[1]
  Gx[nx-1, 2*nx-2] = -2*b2*vx[0,nx-1]/dx[1]
  ## (y,x)=(ny-1,0)
  Gx[(ny-1)*nx, (ny-1)*nx+1] = 2*a2*vx[ny-1,0]/dx[1]
  Gx[(ny-1)*nx, (ny-1)*nx] = -2*a2*vx[ny-1,0]/dx[1]
  Gx[(ny-1)*nx, (ny-2)*nx+1] = 2*b2*vx[ny-1,0]/dx[1]
  Gx[(ny-1)*nx, (ny-2)*nx] = -2*b2*vx[ny-1,0]/dx[1]
  ## (y,x)=(ny-1,nx-1)
  Gx[ny*nx-1, ny*nx-1] = 2*a2*vx[ny-1,nx-1]/dx[1]
  Gx[ny*nx-1, ny*nx-2] = -2*a2*vx[ny-1,nx-1]/dx[1]
  Gx[ny*nx-1, (ny-1)*nx-1] = 2*b2*vx[ny-1,nx-1]/dx[1]
  Gx[ny*nx-1, (ny-1)*nx-2] = -2*b2*vx[ny-1,nx-1]/dx[1]
  ## (0,x)
  for x in range(1,nx-1):
      Gx[x, x+1] = a2*vx[0,x]/dx[1]
      Gx[x, x-1] = -a2*vx[0,x]/dx[1]
      Gx[x, nx+x+1] = b2*vx[0,x]/dx[1]
      Gx[x, nx+x-1] = -b2*vx[0,x]/dx[1]
  ## (ny-1,x)
  for x in range(1,nx-1):
      Gx[(ny-1)*nx+x, (ny-1)*nx+x+1] = a2*vx[ny-1,x]/dx[1]
      Gx[(ny-1)*nx+x, (ny-1)*nx+x-1] = -a2*vx[ny-1,x]/dx[1]
      Gx[(ny-1)*nx+x, (ny-2)*nx+x+1] = b2*vx[ny-1,x]/dx[1]
      Gx[(ny-1)*nx+x, (ny-2)*nx+x-1] = -b2*vx[ny-1,x]/dx[1]
  ## (y,0)
  for y in range(1,ny-1):
      Gx[y*nx, y*nx+1] = 2*a*vx[y,0]/dx[1]
      Gx[y*nx, y*nx] = -2*a*vx[y,0]/dx[1]
      Gx[y*nx, (y-1)*nx+1] = 2*b*vx[y,0]/dx[1]
      Gx[y*nx, (y-1)*nx] = -2*b*vx[y,0]/dx[1]
      Gx[y*nx, (y+1)*nx+1] = 2*b*vx[y,0]/dx[1]
      Gx[y*nx, (y+1)*nx] = -2*b*vx[y,0]/dx[1]
  ## (y,nx-1)
  for y in range(1,ny-1):
      Gx[y*nx+nx-1, y*nx+nx-1] = 2*a*vx[y,nx-1]/dx[1]
      Gx[y*nx+nx-1, y*nx+nx-2] = -2*a*vx[y,nx-1]/dx[1]
      Gx[y*nx+nx-1, (y-1)*nx+nx-1] = 2*b*vx[y,nx-1]/dx[1]
      Gx[y*nx+nx-1, (y-1)*nx+nx-2] = -2*b*vx[y,nx-1]/dx[1]
      Gx[y*nx+nx-1, (y+1)*nx+nx-1] = 2*b*vx[y,nx-1]/dx[1]
      Gx[y*nx+nx-1, (y+1)*nx+nx-2] = -2*b*vx[y,nx-1]/dx[1]
  for y in range(1,ny-1):
    for x in range(1,nx-1):
      Gx[y*nx+x, y*nx+x+1] = a*vx[y,x]/dx[1]
      Gx[y*nx+x, y*nx+x-1] = -a*vx[y,x]/dx[1]
      Gx[y*nx+x, (y-1)*nx+x+1] = b*vx[y,x]/dx[1]
      Gx[y*nx+x, (y+1)*nx+x+1] = b*vx[y,x]/dx[1]
      Gx[y*nx+x, (y-1)*nx+x-1] = -b*vx[y,x]/dx[1]
      Gx[y*nx+x, (y+1)*nx+x-1] = -b*vx[y,x]/dx[1]

  ## grad_y * vy
  Gy = sparse.dok_matrix(((nx)*(ny)+len(constraint_idx),(nx)*(ny)))
  ## (y,x)=(0,0)
  Gy[0, nx] = 2*a2*vy[0,0]/dx[0]
  Gy[0, 0] = -2*a2*vy[0,0]/dx[0]
  Gy[0, nx+1] = 2*b2*vy[0,0]/dx[0]
  Gy[0, 1] = -2*b2*vy[0,0]/dx[0]
  ## (y,x)=(ny-1,0)
  Gy[(ny-1)*nx, (ny-1)*nx] = 2*a2*vy[ny-1,0]/dx[0]
  Gy[(ny-1)*nx, (ny-2)*nx] = -2*a2*vy[ny-1,0]/dx[0]
  Gy[(ny-1)*nx, (ny-1)*nx+1] = 2*b2*vy[ny-1,0]/dx[0]
  Gy[(ny-1)*nx, (ny-2)*nx+1] = -2*b2*vy[ny-1,0]/dx[0]
  ## (y,x)=(0,nx-1)
  Gy[nx-1, nx-1] = -2*a2*vy[0,nx-1]/dx[0]
  Gy[nx-1, 2*nx-1] = 2*a2*vy[0,nx-1]/dx[0]
  Gy[nx-1, nx-2] = -2*b2*vy[0,nx-1]/dx[0]
  Gy[nx-1, 2*nx-2] = 2*b2*vy[0,nx-1]/dx[0]
  ## (y,x)=(ny-1,nx-1)
  Gy[ny*nx-1, ny*nx-1] = 2*a2*vy[ny-1,nx-1]/dx[0]
  Gy[ny*nx-1, (ny-1)*nx-1] = -2*a2*vy[ny-1,nx-1]/dx[0]
  Gy[ny*nx-1, ny*nx-2] = 2*b2*vy[ny-1,nx-1]/dx[0]
  Gy[ny*nx-1, (ny-1)*nx-2] = -2*b2*vy[ny-1,nx-1]/dx[0]
  ## (y,x)=(y,0)
  for y in range(1,ny-1):
      Gy[y*nx, (y+1)*nx] = a2*vy[y,0]/dx[0]
      Gy[y*nx, (y-1)*nx] = -a2*vy[y,0]/dx[0]
      Gy[y*nx, (y+1)*nx+1] = b2*vy[y,0]/dx[0]
      Gy[y*nx, (y-1)*nx+1] = -b2*vy[y,0]/dx[0]
  ## (y,x)=(y,nx-1)
  for y in range(1,ny-1):
      Gy[y*nx+nx-1, (y+1)*nx+nx-1] = a2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y-1)*nx+nx-1] = -a2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y+1)*nx+nx-2] = b2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y-1)*nx+nx-2] = -b2*vy[y,nx-1]/dx[0]
  ## (0,x)
  for x in range(1,nx-1):
      Gy[x, nx+x] = 2*a*vy[0,x]/dx[0]
      Gy[x, x] = -2*a*vy[0,x]/dx[0]
      Gy[x, nx+x+1] = 2*b*vy[0,x]/dx[0]
      Gy[x, x+1] = -2*b*vy[0,x]/dx[0]
      Gy[x, nx+x-1] = 2*b*vy[0,x]/dx[0]
      Gy[x, x-1] = -2*b*vy[0,x]/dx[0]
  ## (ny-1,x)
  for x in range(1,nx-1):
      Gy[(ny-1)*nx+x, (ny-1)*nx+x] = 2*a*vy[ny-1,x]/dx[0]
      Gy[(ny-1)*nx+x, (ny-2)*nx+x] = -2*a*vy[ny-1,x]/dx[0]
      Gy[(ny-1)*nx+x, (ny-1)*nx+x+1] = 2*b*vy[ny-1,x]/dx[0]
      Gy[(ny-1)*nx+x, (ny-2)*nx+x+1] = -2*b*vy[ny-1,x]/dx[0]
      Gy[(ny-1)*nx+x, (ny-1)*nx+x-1] = 2*b*vy[ny-1,x]/dx[0]
      Gy[(ny-1)*nx+x, (ny-2)*nx+x-1] = -2*b*vy[ny-1,x]/dx[0]
  for y in range(1,ny-1):
      Gy[y*nx+nx-1, (y+1)*nx+nx-1] = a2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y-1)*nx+nx-1] = -a2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y+1)*nx+nx-2] = b2*vy[y,nx-1]/dx[0]
      Gy[y*nx+nx-1, (y-1)*nx+nx-2] = -b2*vy[y,nx-1]/dx[0]
  for y in range(1,ny-1):
    for x in range(1,nx-1):
      Gy[y*nx+x, (y+1)*nx+x] = a*vy[y,x]/dx[0]
      Gy[y*nx+x, (y-1)*nx+x] = -a*vy[y,x]/dx[0]
      Gy[y*nx+x, (y+1)*nx+x+1] = b*vy[y,x]/dx[0]
      Gy[y*nx+x, (y-1)*nx+x+1] = -b*vy[y,x]/dx[0]
      Gy[y*nx+x, (y+1)*nx+x-1] = b*vy[y,x]/dx[0]
      Gy[y*nx+x, (y-1)*nx+x-1] = -b*vy[y,x]/dx[0]

  #return(Gx.todense(), Gy.todense())
  return(Gx.tocsr() + Gy.tocsr())

