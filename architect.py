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
        # self.optimizer = torch.optim.Adam([{'params': self.model.alpha_parameters(), 'lr': 0},
        #                                    {'params': self.model.beta_parameters(), 'lr': args.arch_learning_rate}],
        #                                   betas=(0.5, 0.999),
        #                                   weight_decay=args.arch_weight_decay)

    def step(self, input_valid, target_valid, epoch):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, epoch)
        self.optimizer.step()

    # def set(self):
    #     for i, param_group in enumerate(self.optimizer.param_groups):
    #         if i == 0:
    #             param_group['lr'] = self.lr_alpha
    
    def _backward_step(self, input_valid, target_valid, epoch):
        normal_alphas = F.softmax(self.model.arch_parameters()[0], dim=-1)
        reduce_alphas = F.softmax(self.model.arch_parameters()[1], dim=-1)

        avg = 0.0
        alpha_loss1 = -torch.sum((1.0-normal_alphas)**2)
        alpha_loss2 = -torch.sum((1.0-reduce_alphas)**2)
        # alpha_loss = ent_normal + ent_reduce
        # ent_normal = -torch.sum(torch.mul(normal_alphas, torch.log(normal_alphas)))
        # ent_reduce = -torch.sum(torch.mul(reduce_alphas, torch.log(reduce_alphas)))
        # alpha_loss = torch.add(ent_normal, ent_reduce)

        # normal_betas1 = F.softmax(self.model.arch_parameters()[2][0:3])
        # aa1 = torch.zeros_like(normal_betas1)
        # _,n1 = normal_betas1.topk(2)
        # aa1[n1] = 0.5
        # normal_betas2 = F.softmax(self.model.arch_parameters()[2][3:7])
        # aa2 = torch.zeros_like(normal_betas2)
        # _,n2 = normal_betas2.topk(2)
        # aa2[n2] = 0.5
        # normal_betas3 = F.softmax(self.model.arch_parameters()[2][7:12])
        # aa3 = torch.zeros_like(normal_betas3)
        # _,n3 = normal_betas3.topk(2)
        # aa3[n3] = 0.5
        # reduce_betas1 = F.softmax(self.model.arch_parameters()[3][0:3])
        # bb1 = torch.zeros_like(reduce_betas1)
        # _,r1 = reduce_betas1.topk(2)
        # bb1[r1] = 0.5
        # reduce_betas2 = F.softmax(self.model.arch_parameters()[3][3:7])
        # bb2 = torch.zeros_like(reduce_betas2)
        # _,r2 = reduce_betas2.topk(2)
        # bb2[r2] = 0.5
        # reduce_betas3 = F.softmax(self.model.arch_parameters()[3][7:12])
        # bb3 = torch.zeros_like(reduce_betas3)
        # _,r3 = reduce_betas3.topk(2)
        # bb3[r3] = 0.5

        # beta_loss1 = torch.sum(torch.square(normal_betas1 - aa1)) + torch.sum(torch.square(normal_betas2 - aa2)) + torch.sum(torch.square(normal_betas3 - aa3))
        # beta_loss2 = torch.sum(torch.square(reduce_betas1 - bb1)) + torch.sum(torch.square(reduce_betas2 - bb2)) + torch.sum(torch.square(reduce_betas3 - bb3))

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

def exp(epoch, e, Epochs):
    if epoch < e:
        out = 0
    else:
        out = math.exp((epoch-Epochs+1)/5)
    return out

def log(epoch, e, Epochs):
    if epoch < e:
        out = 0
    else:
        out = math.log(epoch-e+1, Epochs-e)
    return out
