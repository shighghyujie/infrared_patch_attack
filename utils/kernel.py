import torch

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def kernel_3x3():
    kernel = torch.ones((3,3)) 
    kernel[0][0] = kernel[0][2] = kernel[2][0] = kernel[2][2] = 1/4
    kernel[0][1] = kernel[1][0] = kernel[1][2] = kernel[2][1] = 1/2
    kernel[1][1] = 0
    kernel = torch.FloatTensor(kernel)
    kernel = torch.stack([kernel]) 
    kernel = kernel.unsqueeze(0).to(device) 
    return kernel  

def kernel_5x5():
    kernel = torch.ones((5,5))  
    kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 7
    kernel[0][1] = kernel[0][3] = kernel[1][0] = kernel[1][4] = kernel[3][0] = kernel[3][4] = kernel[4][1] = kernel[4][3] = 10
    kernel[0][2] = kernel[2][0] = kernel[2][4] = kernel[4][2] = 13
    kernel[1][1] = kernel[1][3] = kernel[3][1] = kernel[3][3] = 14
    kernel[1][2] = kernel[2][1] = kernel[2][3] = kernel[3][2] = 18
    kernel[2][2] = 0
    kernel = torch.FloatTensor(kernel)
    kernel = torch.stack([kernel])
    kernel = kernel.unsqueeze(0).to(device)
    return kernel
