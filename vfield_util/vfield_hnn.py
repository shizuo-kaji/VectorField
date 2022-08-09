## Approximating Hamiltonian from partially observed vector field data
## by Shizuo KAJI

import numpy as np
import skimage.morphology
import os
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,sobel
import cv2
import torch
import torch.nn as nn


## compute the normal direction of the region X using image processing tech: gradient of gaussian blur
def boundary_normal(img, sigma=5):
    image, contours = cv2.findContours(img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #plt.imshow(X)
    Y=gaussian_filter(img.astype(np.float64), sigma=sigma)
    #print(image[0])
    cc = np.array([c for c in image[0].reshape(-1,2) if 0<c[0]<img.shape[1]-1 and 0<c[1]<img.shape[0]-1],dtype=np.int32)
    sx = sobel(Y, axis=1,mode="constant")
    sy = sobel(Y, axis=0,mode="constant")
    #sy,sx = np.gradient(Y)
    #mgn = np.linalg.norm(v, axis=1)
    mgn = np.sqrt(sx**2+sy**2)
    sx /= np.maximum(mgn,1e-10)
    sy /= np.maximum(mgn,1e-10)
    if len(cc)==0:
        cc = np.zeros((0,2))
        sx = np.zeros((0,0))
        sy = np.zeros((0,0))
    else:
        sx = sx[cc[:,1],cc[:,0]]
        sy = sy[cc[:,1],cc[:,0]]
    return(cc,sx,sy)


## DL model for approximating the hamiltonian
class HNN(torch.nn.Module):
    def __init__(self,  input_dim=2, hidden_dim=200, output_dim=1):
        super(HNN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight) # use a principled initialization

        # complex form
        self.J = torch.tensor([[0,1],[-1,0]], dtype=torch.float)
        
    def forward(self, x):
        h = torch.relu( self.linear1(x) )
        h = torch.relu( self.linear2(h) )
        return self.linear3(h)

    def vfield(self, x):
        H = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        return(dH @ self.J)
    
    
    
# train the model
def fit_HNN(X,Y,vx,vy,bd_x,bd_y,bd_vx,bd_vy,epochs = 10000,batch_size=None,lr = 1e-3,lambda_boundary = 1.0, hidden_dim=100, device=None):
    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_list = torch.tensor( np.stack([X,Y],axis=-1), requires_grad=True, dtype=torch.float32).to(device)
    v_list = torch.Tensor(np.stack([vx,vy],axis=-1)).to(device)
    #h_list = torch.Tensor(H[mask]).to(device)
    bd_x = torch.tensor(np.stack([bd_x,bd_y],axis=-1), requires_grad=True, dtype=torch.float32).to(device)
    bd_n = torch.Tensor(np.stack([bd_vx,bd_vy],axis=-1)).to(device)
    print(f"#boundary pts {len(bd_x)}, #region pts {len(x_list)}")

    if batch_size is None:
        batch_size = X.size 
    model = HNN(hidden_dim=hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=0)
    loss_func = nn.MSELoss(reduction='mean').to(device)

    for step in range(epochs):
        ixs = torch.randperm(x_list.shape[0])[:batch_size]
        v_estimated = model.vfield(x_list[ixs])
        loss = loss_func(v_list[ixs], v_estimated)
        #print(bd_n.shape,(torch.sum(bd_v_estimated*bd_n, dim=1)**2).sum())

        ## slip boundary
        if lambda_boundary>0:
            bd_v_estimated = model.vfield(bd_x)
            bd_loss = (torch.sum(bd_v_estimated*bd_n, dim=1)**2).sum()
            loss += lambda_boundary*bd_loss

        ## directly with H    
        #H_estimated = model(x[ixs])
        #loss = loss_func(h[ixs], H_estimated)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % (epochs//10) == 0:
            print(f'iter {step}, loss {loss.item()}, boundary loss {bd_loss.item()}')
    return(model)