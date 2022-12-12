import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

OPS = {
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'bin_conv_3x3' : lambda C, stride, affine: BinReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'bin_conv_5x5' : lambda C, stride, affine: BinReLUConvBN(C, C, 5, stride, 2, affine=affine),
  'bin_dil_conv_3x3' : lambda C, stride, affine: BinDilConv(C, C, 3, stride, 2, 2, affine=affine),
  'bin_dil_conv_5x5' : lambda C, stride, affine: BinDilConv(C, C, 5, stride, 4, 2, affine=affine)
}


class BinaryActivation(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 - 2 * torch.abs(input)
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class BinaryWeight(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class myConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(myConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.mean(torch.mean(torch.mean(abs(bw), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True).detach()
        bw = BinaryWeight().apply(w)
        bw = bw * sw
        output = F.conv2d(x, bw, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=self.bias)
        return output


class BinReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(BinReLUConvBN, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
        # self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, groups=25, bias=False)
        self.conv = myConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.prelu = nn.PReLU(C_out)

    def forward(self, x):
        res = x
        x = self.bn1(x)
        x = BinaryActivation().apply(x)
        x = self.conv(x)
        x = self.bn2(x)
        
        if self.stride == 2:
            res = self.pooling(res)
        x += res
        x = self.prelu(x)
        return x


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
        # self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv = myConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)
        self.prelu = nn.PReLU(C_out)

    def forward(self, x):
        x = self.bn1(x)
        x = BinaryActivation().apply(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.prelu(x)
        return x


class BinDilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(BinDilConv, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
        # self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=25, bias=False)
        self.conv = myConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.prelu = nn.PReLU(C_out)

    def forward(self, x):
        res = x
        x = self.bn1(x)
        x = BinaryActivation().apply(x)
        x = self.conv(x)
        x = self.bn2(x)
        
        if self.stride == 2:
            res = self.pooling(res)
        x += res
        x = self.prelu(x)
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)
        # self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_1 = myConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = myConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)
        self.prelu = nn.PReLU(C_out)

    def forward(self, x):
        x = self.bn1(x)
        x = BinaryActivation().apply(x)
        x = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        x = self.bn2(x)
        x = self.prelu(x)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

