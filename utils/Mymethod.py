import torch
import torch.nn as nn
import torch.nn.functional as F

class MyThresholdMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        alpha = 1
        ctx.save_for_backward(input)
        return input.clamp(max=alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 1
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > alpha] = 0
        return grad_input

class thredOne(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        alpha = 0.2
        one = torch.ones_like(input)
        input = torch.where(input < alpha, one, input)
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 0.2
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < alpha] = 0
        return grad_input

class grad_modify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 0.1
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        padding = nn.ZeroPad2d((1, 1, 1, 1))
        mask_padding = padding(input)
        kernel = torch.ones((3,3))
        kernel = kernel/9
        kernel[0][0] = kernel[0][2] = kernel[2][0] = kernel[2][2] = 1/9
        kernel[0][1] = kernel[1][0] = kernel[1][2] = kernel[2][1] = 1/4
        kernel[1][1] = 1/2
        kernel = torch.FloatTensor(kernel)
        kernel = torch.stack([kernel])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        kernel = kernel.unsqueeze(0).to(device)
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1)
        grad_input = grad_input*(msk/msk.max())
        return grad_input