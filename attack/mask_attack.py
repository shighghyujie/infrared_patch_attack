import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from yolov3.detect import detect_train
from utils.kernel import kernel_3x3, kernel_5x5
from utils import MyThresholdMethod, thredOne, grad_modify
from torch.autograd import Variable
import scipy.stats as st
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

trans = transforms.Compose([
                transforms.ToTensor(),
            ])
# grad_avg TI 参数设置
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels = 3                                      # 3通道
kernel_size = 3                                  # kernel大小
kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
gaussian_kernel = np.stack([kernel])   # 5*5*3
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda

inputsize = [416,416]

def shaped_mask_attack(H, W, bbox, model, img, device, emp_iterations, max_pertubation_mask = 100, content = 0, lambda_sparse=5, lambda_attack=20, lambda_agg=5, grad_avg=False):
    ## 图片预处理 ##
    X_ori =  torch.stack([trans(img)]).to(device) 
    X_ori = F.interpolate(X_ori, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
    
    ## 随机生成mask ##
    objbox = torch.rand((H, W)).to(device)    
    mask = Variable(objbox, requires_grad=True)
    ## 应用自定义方法类 ##
    threM = MyThresholdMethod.apply
    threone = thredOne.apply
    gradmodify = grad_modify.apply
    grad_momentum = 0
    
    ## 迭代生成对抗样本 ##
    for itr in range(emp_iterations):  
        mask_extrude = mask 
        mask_extrude = mask_extrude ** 2 / (mask_extrude ** 2).sum() * max_pertubation_mask # 限制mask的范围   
        # mask_extrude = mask_extrude / mask_extrude.sum() * max_pertubation_mask # 限制mask的范围   
        mask_extrude = threM(mask_extrude) # mask中大于1的值置0
        mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
        mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
        mask_modify = gradmodify(mask_extrude)
        mask_resize = nn.functional.interpolate(mask_modify, (bbox[1] - bbox[0], bbox[3] - bbox[2]), mode='bilinear', align_corners=False)
        # pad
        padding = nn.ZeroPad2d((bbox[2], 416-bbox[3], bbox[0], 416-bbox[1]))
        mask_pad = padding(mask_resize)
        X_adv_b = X_ori * (1 - mask_pad) + content * mask_pad # 生成计算损失用的对抗样本
        
        # 攻击损失 #
        loss_attack = detect_train(model, X_adv_b) 
        
        # 值稀疏正则项 #
        m = threone(mask_extrude)
        o = torch.ones_like(m)
        loss_sparse = -F.mse_loss(m, o) * 100 + (mask_extrude[0][0] ** 4).sum() / max_pertubation_mask 
        
        # 集聚正则项 #
        padding = nn.ZeroPad2d((2, 2, 2, 2)) # 上下左右均添加2dim
        mask_padding = padding(mask_extrude) # 对mask_extrude进行填充
        kernel = kernel_5x5() 
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        loss_agg = ((msk)*mask_extrude).sum()

        padding = nn.ZeroPad2d((1, 1, 1, 1))
        mask_padding = padding(mask_extrude)
        kernel = kernel_3x3() 
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        loss_agg2 = ((msk)*mask_extrude).sum()
        num_nozero = torch.sum(mask_extrude!=0)
        # 总损失函数 #
        loss_total =  loss_sparse * lambda_sparse + loss_attack * lambda_attack + loss_agg / lambda_agg
        
        loss_total.backward()

        # 带动量的SGD优化 #
        grad_c = mask.grad.clone()
        grad_c = grad_c.reshape(1, 1, grad_c.shape[0], grad_c.shape[1])
        if grad_avg:
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(1,1), groups=1)[0][0]
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.85 * grad_momentum   # 1
        grad_momentum = grad_a     
        mask.grad.zero_()
        mask.data = mask.data + 0.1 * torch.sign(grad_momentum)
        mask.data = mask.data.clamp(0., 1.)

    # print("aggregation:", loss_agg2 * 8 / 28 / num_nozero)
    # print("aggregation:", loss_agg.item())
    # print(num_nozero)
    ## 利用生成的mask生成攻击后的图片 ##
    one = torch.ones_like(mask_pad)
    zero = torch.zeros_like(mask_pad)
    mask_extrude = torch.where(mask_pad > 0.1, one, zero)
    X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
    adv_face_ts = X_adv.cpu().detach()
    adv_final = X_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_img = Image.fromarray(adv_x_255)
    mask = mask_extrude.cpu().detach()[0][0].numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    return adv_face_ts, adv_img, mask


# def shaped_mask_attack(H, W, bbox, model, img, device, emp_iterations, max_pertubation_mask = 100, content = 0, lambda_sparse=5, lambda_attack=20, lambda_agg=25):
#     ## 图片预处理 ##
#     X_ori =  torch.stack([trans(img)]).to(device) 
#     X_ori = F.interpolate(X_ori, (H, W), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
    
#     ## 随机生成mask, 但检测框外的部分值全设为0 ##
#     mask = torch.rand_like(X_ori[0][0], requires_grad=True).to(device)  
#     facemask = torch.zeros((H, W)).to(device) 
#     facemask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = torch.ones(((bbox[3]-bbox[1],bbox[2]-bbox[0]))) 
#     mask.data = mask.data * facemask
    
#     ## 应用自定义方法类 ##
#     threM = MyThresholdMethod.apply
#     threone = thredOne.apply
#     gradmodify = grad_modify.apply
#     grad_momentum = 0
    
#     ## 迭代生成对抗样本 ##
#     for itr in range(emp_iterations):  
#         mask_extrude = mask 
#         mask_extrude = mask_extrude ** 2 / (mask_extrude ** 2).sum() * max_pertubation_mask # 限制mask的范围   
#         # mask_extrude = mask_extrude / mask_extrude.sum() * max_pertubation_mask # 限制mask的范围   
#         mask_extrude = mask_extrude * facemask # mask作用facemask，只在facemask为1的时候有作用
#         mask_extrude = threM(mask_extrude) # mask中大于1的值置0
#         mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
#         mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
#         X_adv_b = X_ori * (1 - gradmodify(mask_extrude)) + content * gradmodify(mask_extrude) # 生成计算损失用的对抗样本
        
#         # 攻击损失 #
#         loss_attack = detect_train(model, X_adv_b) 
        
#         # 值稀疏正则项 #
#         m = threone(mask_extrude)
#         o = torch.ones_like(m)
#         loss_sparse = -F.mse_loss(m, o) * 100 + (mask_extrude[0][0] ** 4).sum() / max_pertubation_mask 
        
#         # 集聚正则项 #
#         padding = nn.ZeroPad2d((2, 2, 2, 2)) # 上下左右均添加2dim
#         mask_padding = padding(mask_extrude) # 对mask_extrude进行填充
#         kernel = kernel_5x5() 
#         msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
#         loss_agg = ((msk)*mask_extrude).sum()

#         padding = nn.ZeroPad2d((1, 1, 1, 1))
#         mask_padding = padding(mask_extrude)
#         kernel = kernel_3x3() 
#         msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
#         loss_agg2 = ((msk)*mask_extrude).sum()
#         num_nozero = torch.sum(mask_extrude!=0)
#         # 总损失函数 #
#         loss_total =  loss_sparse * lambda_sparse + loss_attack * lambda_attack + loss_agg / lambda_agg
        
#         loss_total.backward()
#         # print loss
#         # l1 = -loss_attack.item()*20
#         # l2 = -(mask_extrude[0][0] ** 4).sum().item()*10
#         # l3 = -loss_agg2.item()
#         # print_loss = l1+l2+l3
#         # print("loss: {}".format(print_loss))
        

#         # 带动量的SGD优化 #
#         grad_c = mask.grad.clone()
#         grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.85 * grad_momentum   # 1
#         grad_momentum = grad_a     
#         mask.grad.zero_()
#         mask.data = mask.data + 0.1 * torch.sign(grad_momentum)
#         mask.data = mask.data.clamp(0., 1.)

#     # print("aggregation:", loss_agg2 * 8 / 28 / num_nozero)
#     # print("aggregation:", loss_agg.item())
#     # print(num_nozero)
#     ## 利用生成的mask生成攻击后的图片 ##
#     one = torch.ones_like(mask_extrude)
#     zero = torch.zeros_like(mask_extrude)
#     mask_extrude = torch.where(mask_extrude > 0.1, one, zero)
#     X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
#     adv_face_ts = X_adv.cpu().detach()
#     adv_final = X_adv[0].cpu().detach().numpy()
#     adv_final = (adv_final * 255).astype(np.uint8)
#     adv_x_255 = np.transpose(adv_final, (1, 2, 0))
#     adv_img = Image.fromarray(adv_x_255)
#     return adv_face_ts, adv_img, mask_extrude.cpu().detach()


    
