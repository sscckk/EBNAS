import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from genotypes import PRIMITIVES


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.lr_alpha = args.arch_learning_rate
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)


    def step(self, input_valid, target_valid, epoch):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, epoch)
        self.optimizer.step()

    
    def _backward_step(self, input_valid, target_valid, epoch):
        normal_alphas = F.softmax(self.model.arch_parameters()[0], dim=-1)
        reduce_alphas = F.softmax(self.model.arch_parameters()[1], dim=-1)

        alpha_loss1 = -torch.sum((1.0-normal_alphas)**2)
        alpha_loss2 = -torch.sum((1.0-reduce_alphas)**2)

        normal_betas1 = torch.sigmoid(self.model.arch_parameters()[2][0:3]+0.6931)
        normal_betas2 = torch.sigmoid(self.model.arch_parameters()[2][3:7])
        normal_betas3 = torch.sigmoid(self.model.arch_parameters()[2][7:12]-0.4055)
        reduce_betas1 = torch.sigmoid(self.model.arch_parameters()[3][0:3]+0.6931)
        reduce_betas2 = torch.sigmoid(self.model.arch_parameters()[3][3:7])
        reduce_betas3 = torch.sigmoid(self.model.arch_parameters()[3][7:12]-0.4055)

        beta1_var = torch.sum(torch.square(normal_betas1-torch.mean(normal_betas1)))
        beta2_var = torch.sum(torch.square(normal_betas2-torch.mean(normal_betas2)))
        beta3_var = torch.sum(torch.square(normal_betas3-torch.mean(normal_betas3)))
        beta4_var = torch.sum(torch.square(reduce_betas1-torch.mean(reduce_betas1)))
        beta5_var = torch.sum(torch.square(reduce_betas2-torch.mean(reduce_betas2)))
        beta6_var = torch.sum(torch.square(reduce_betas3-torch.mean(reduce_betas3)))

        beta_loss1 = -(beta1_var + beta2_var + beta3_var) + 0.5 * (torch.square(torch.sum(normal_betas1) - 2) + torch.square(torch.sum(normal_betas2) - 2) + torch.square(torch.sum(normal_betas3) - 2))

        beta_loss2 = -(beta4_var + beta5_var + beta6_var) + 0.5 * (torch.square(torch.sum(reduce_betas1) - 2) + torch.square(torch.sum(reduce_betas2) - 2) + torch.square(torch.sum(reduce_betas3) - 2))

        loss = self.model._loss(input_valid, target_valid) + 1.5 * linear(epoch,0,50) * (alpha_loss1 + alpha_loss2) + 0.8 * linear(epoch,0,50) * (beta_loss1 + beta_loss2)  
        loss.backward()


def linear(epoch, e, Epochs):
    if epoch < e:
        out = 0
    else:
        out = (epoch-e) / (Epochs-e-1)
    return out
