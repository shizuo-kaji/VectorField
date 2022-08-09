## Various utility functions related to vector field analysis
## By Shizuo KAJI

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate,sparse
import matplotlib.colors as colors
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix

# Helmholtz decomposition by convolving the Green function for the free boundary
def decomp(vfield): 
  kx = np.fft.fftfreq(vfield.shape[0]).reshape(-1,1)
  ky = np.fft.fftfreq(vfield.shape[1])
  k2 = kx**2 + ky**2
  k2[0,0] = 1.
  dk = (np.fft.fftn(vfield[:,:,0]) * kx +  np.fft.fftn(vfield[:,:,1]) * ky) / k2
  r = np.stack((np.fft.ifftn(dk * kx).real, np.fft.ifftn(dk * ky).real), axis=-1) # div free
  d = np.stack((vfield[:,:,0] - r[:,:,0], vfield[:,:,1] - r[:,:,1]), axis=-1 ) # rot free
  # potentials
  h = integrate2(np.stack((r[:,:,1],-r[:,:,0]), axis=-1))
  g = integrate2(d)
  return(r,d,h,g)

# gradient
def grad(f,dx=(1,1)):
  ddy, ddx = np.gradient(f, dx[0], dx[1])
  return(np.stack((ddx, ddy), axis = -1))

# rotated gradient
def rot_grad(f,dx=(1,1)):
  ddy, ddx = np.gradient(f, dx[0], dx[1])
  ddy *= -1.0
  return(np.stack((ddy, ddx), axis=-1))

# divergence and curl
def divcurl(vfield, dx=(1,1)):
  dudy, dudx = np.gradient(vfield[:,:,0], dx[0], dx[1])
  dvdy, dvdx = np.gradient(vfield[:,:,1], dx[0], dx[1])
  np.add(dudx, dvdy, dudx)
  np.subtract(dvdx, dudy, dvdx)
  return (dudx, dvdx)

# quick-dirty integrator of a 2d vector field
def integrate(v, dx=(1,1)): 
  SdZx = np.cumsum(v[:,:,0], axis=1)*dx[1]
  SdZy = np.cumsum(v[:,:,1], axis=0)*dx[0]
  N,M = SdZx.shape
  Zhat = np.zeros((N,M))
  for i in range(N):
      for j in range(M):
          Zhat[i,j] = SdZx[0,M//2] +SdZy[i,M//2]+ SdZx[i,j]-SdZx[i,M//2]
  return(Zhat)

# integration after interpolation of a 2d vector field
def integrate2(v, fac=2, dx=(1,1)): 
  M, N = v.shape[:2]
  xl = np.linspace(0, 1, N)
  yl = np.linspace(0, 1, M)
  X, Y = np.meshgrid(xl, yl)
  r = np.stack([X.ravel(), Y.ravel()]).T
  Sx = interpolate.CloughTocher2DInterpolator(r, v[:,:,0].ravel())
  Sy = interpolate.CloughTocher2DInterpolator(r, v[:,:,1].ravel())
  # slow
#  Sx = interpolate.Rbf(r[:,0], r[:,1], v[:,:,0].ravel(), function='thin_plate')
#  Sy = interpolate.Rbf(r[:,0], r[:,1], v[:,:,1].ravel(), function='thin_plate')
  xli = np.linspace(0, 1, fac*N)
  yli = np.linspace(0, 1, fac*M)
  Xi, Yi = np.meshgrid(xli, yli)
  ri = np.stack([Xi.ravel(), Yi.ravel()]).T
  dZdxi = Sx(ri).reshape(Xi.shape)
  dZdyi = Sy(ri).reshape(Xi.shape)
#  dZdxi = Sx(ri[:,0],ri[:,1]).reshape(Xi.shape)
#  dZdyi = Sy(ri[:,0],ri[:,1]).reshape(Xi.shape)
#  print(np.abs(v[:,:,0]-dZdxi[::fac,::fac]).mean())
  SdZxi = np.nancumsum(dZdxi, axis=1)*dx[1]/fac
  SdZyi = np.nancumsum(dZdyi, axis=0)*dx[0]/fac
  Zhati = np.zeros(SdZxi.shape)
  N, M = Zhati.shape
  for i in range(N):
      for j in range(M):
          Zhati[i,j] = SdZxi[0,M//2] +SdZyi[i,M//2]+ SdZxi[i,j]-SdZxi[i,M//2]
#          Zhati[i,j] = SdZyi[i,0]-SdZyi[0,0]+SdZxi[i,j]-SdZxi[i,0]
  return(Zhati[::fac,::fac])

## interpolation of a sparsely sampled vector field by RBFs
def basis_func(point, centre, sigma=1):  # Gaussian RBF
    result = np.exp(-np.sum((point-centre) ** 2,axis=1)/sigma)
    return result

def interpolate_vfield(sample_x, sample_y, sample_vx, sample_vy, min_x=0, max_x=0, min_y=0, max_y=0, NX=10, NY=10, neighbours=5, sigma=1):
    if min_x==max_x:
        min_x = np.min(sample_x)
        max_x = np.max(sample_x)
    if min_y==max_y:
        min_y = np.min(sample_y)
        max_y = np.max(sample_y)
    rbf_x, rbf_y = np.meshgrid(np.linspace(min_x, max_x, NX), np.linspace(min_y, max_y, NY))
    rbf_c = np.vstack([rbf_x.ravel(), rbf_y.ravel()]).T
    #print(rbf_c)
    
    A=np.zeros((len(sample_x), len(rbf_c))) 
    for i in range(len(sample_x)):
        xy = np.array((sample_x[i],sample_y[i]))
        col_dist = np.sqrt(np.sum((rbf_c-xy)**2,axis=1))
        dist_sort_index = col_dist.argsort()
        neighbour_index = dist_sort_index[:neighbours] # pick nearest neibours from the constrained sample point
        A[i,neighbour_index] = basis_func(xy, rbf_c[neighbour_index], sigma)
        #print(xy,neighbour_index)

    #print(A)
    A = csc_matrix(A)
    weight_x = lsqr(A, sample_vx)[0]
    weight_y = lsqr(A, sample_vy)[0]
    #print(weight_x, weight_y)
    #print(sample_vx, sample_vy)

    def interpolant(x,y):
        xy = np.array((x,y))
        col_dist = np.sqrt(np.sum((rbf_c-xy)**2,axis=1))
        dist_sort_index = col_dist.argsort()
        neighbour_index = dist_sort_index[:neighbours]
        wx = np.zeros_like(weight_x)
        wy = np.zeros_like(weight_y)
        wx[neighbour_index] = weight_x[neighbour_index]
        wy[neighbour_index] = weight_y[neighbour_index]
        vx = np.sum(wx*basis_func(xy,rbf_c,sigma))
        vy = np.sum(wy*basis_func(xy,rbf_c,sigma))
        return(vx,vy)

    return(interpolant)

## sample vectors from an interpolated vector field
def sampling(X,Y,interpolation_function):
    vx = np.zeros_like(X)
    vy = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vx[i,j], vy[i,j] = interpolation_function(X[i,j],Y[i,j])
            #print(X[i,j],Y[i,j],vx[i,j], vy[i,j])
    return(vx,vy)

