import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import copy


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, k):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2,2)
        self.k = k
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // self.k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[ : , : dim_2//self.k, :, :]
        xtemp2 = x[ : , dim_2//self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        return ans


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                k = 4 if reduction else 8
                op = MixedOp(C, stride, k)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                
                weights2 = torch.ones(2, requires_grad=False).cuda()
                tw23 = torch.sigmoid(self.betas_reduce[0:3]+0.6931)
                tw24 = torch.sigmoid(self.betas_reduce[3:7])
                tw25 = torch.sigmoid(self.betas_reduce[7:12]-0.4055)
                weights2 = torch.cat([weights2,tw23,tw24,tw25],dim=0)
                # n = 3
                # start = 0
                # for i in range(self._steps-1):
                #     end = start + n
                #     tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                #     start = end
                #     n += 1
                #     weights2 = torch.cat([weights2,tw2],dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                
                weights2 = torch.ones(2, requires_grad=False).cuda()
                tw23 = torch.sigmoid(self.betas_normal[0:3]+0.6931)
                tw24 = torch.sigmoid(self.betas_normal[3:7])
                tw25 = torch.sigmoid(self.betas_normal[7:12]-0.4055)
                weights2 = torch.cat([weights2,tw23,tw24,tw25],dim=0)
                # n = 3
                # start = 0
                # for i in range(self._steps-1):
                #     end = start + n
                #     tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                #     start = end
                #     n += 1
                #     weights2 = torch.cat([weights2,tw2],dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3*torch.randn(k-2).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3*torch.randn(k-2).cuda(), requires_grad=True)

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

        self._alpha_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        self._beta_parameters = [
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters
    
    def alpha_parameters(self):
        return self._alpha_parameters
    
    def beta_parameters(self):
        return self._beta_parameters

    def genotype(self):
    
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        def _sift_beta(betas, W):
            offset = 2
            node3 = sorted(range(len(betas[0:3])), key=lambda x: betas[0:3][x])
            node4 = sorted(range(len(betas[3:7])), key=lambda x: betas[3:7][x])
            node5 = sorted(range(len(betas[7:12])), key=lambda x: betas[7:12][x])
            W[offset + node3[0]][:] = 0
            offset += 3
            W[offset + node4[0]][:] = 0
            W[offset + node4[1]][:] = 0
            offset += 4
            W[offset + node5[0]][:] = 0
            W[offset + node5[1]][:] = 0
            W[offset + node5[2]][:] = 0
            return W

        with torch.no_grad():
            alphas_normal = copy.deepcopy(self.alphas_normal)
            alphas_reduce = copy.deepcopy(self.alphas_reduce)
            alphas_normal = _sift_beta(self.betas_normal, alphas_normal)
            alphas_reduce = _sift_beta(self.betas_reduce, alphas_reduce)

            gene_normal = _parse(F.softmax(alphas_normal, dim=-1).data.cpu().numpy())
            gene_reduce = _parse(F.softmax(alphas_reduce, dim=-1).data.cpu().numpy())

            concat = range(2+self._steps-self._multiplier, self._steps+2)
            genotype = Genotype(
                normal=gene_normal, normal_concat=concat,
                reduce=gene_reduce, reduce_concat=concat
            )
        return genotype


