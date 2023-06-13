#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b7
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

class resnext(nn.Module):
    def __init__(self):
        super(resnext, self).__init__()

        self.model = resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.DEFAULT) 
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        
    def forward(self, x):
        return self.model(x)
class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()

        self.model = efficientnet_b7(pretrained=True) 
        list(self.model.features.children())[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = nn.Sequential(self.model.classifier[0], 
                                            nn.Linear(in_features=2560, out_features=100, bias=True),
                                            nn.Linear(in_features=100, out_features=2, bias=True))
        
    def forward(self, x):
        return self.model(x)


def getSizes(chz, growth, blks=4):
    # This function does not calculate the size requirements for head and tail

    # Encoder sizes
    sizes = {'enc': {'inter':[], 'ip':[], 'op': []},
             'dec': {'skip':[], 'ip': [], 'op': []}}
    sizes['enc']['inter'] = np.array([chz*(i+1) for i in range(0, blks)])
    sizes['enc']['op'] = np.array([int(growth*chz*(i+1)) for i in range(0, blks)])
    sizes['enc']['ip'] = np.array([chz] + [int(growth*chz*(i+1)) for i in range(0, blks-1)])

    # Decoder sizes
    sizes['dec']['skip'] = sizes['enc']['ip'][::-1] + sizes['enc']['inter'][::-1]
    sizes['dec']['ip'] = sizes['enc']['op'][::-1] #+ sizes['dec']['skip']
    sizes['dec']['op'] = np.append(sizes['enc']['op'][::-1][1:], chz)
    return sizes

class convBlock(nn.Module):
    def __init__(self, in_c, inter_c, out_c, actfunc):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inter_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = torch.nn.BatchNorm2d(num_features=out_c)
    def forward(self, x):
        x = self.actfunc(self.conv1(x))
        x = self.actfunc(self.conv2(x)) # Remove x if not working properly
        x = self.bn(x)
        return x
        
class linStack(torch.nn.Module):
    """A stack of linear layers followed by batch norm and hardTanh

    Attributes:
        num_layers: the number of linear layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, bias, actBool, dp):
        super().__init__()

        layers_lin = []
        for i in range(num_layers):
            m = torch.nn.Linear(hidden_dim if i > 0 else in_dim,
                hidden_dim if i < num_layers - 1 else out_dim, bias=bias)
            layers_lin.append(m)
        self.layersLin = torch.nn.ModuleList(layers_lin)
        self.act_func = torch.nn.SELU()
        self.actBool = actBool
        self.dp = torch.nn.Dropout(p=dp)

    def forward(self, x):
        # Input shape (batch, features, *)
        for i, _ in enumerate(self.layersLin):
            x = self.act_func(x) if self.actBool else x
            x = self.layersLin[i](x)
            x = self.dp(x)
        return x

class Transition_down(nn.Module):
    def __init__(self, in_c, out_c, down_size, norm, actfunc):
        super(Transition_down, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.max_pool = nn.AvgPool2d(kernel_size=down_size) if down_size else False
        self.norm = norm(num_features=in_c)
        self.actfunc = actfunc
    def forward(self, x):
        x = self.actfunc(self.norm(x))
        x = self.conv(x)
        x = self.max_pool(x) if self.max_pool else x
        return x

class DenseNet2D_down_block(nn.Module):
    def __init__(self, in_c, inter_c, op_c, down_size, norm, actfunc):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, inter_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_c+inter_c, inter_c, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(in_c+2*inter_c, inter_c, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(inter_c, inter_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.bn = norm(num_features=in_c)
        self.TD = Transition_down(inter_c+in_c, op_c, down_size, norm, actfunc)

    def forward(self, x):
        x1 = self.actfunc(self.conv1(self.bn(x)))
        x21 = torch.cat([x, x1], dim=1)
        x22 = self.actfunc(self.conv22(self.conv21(x21)))
        x31 = torch.cat([x21, x22], dim=1)
        out = self.actfunc(self.conv32(self.conv31(x31)))
        out = torch.cat([out, x], dim=1)
        return out, self.TD(out)

class DenseNet2D_up_block(nn.Module):
    def __init__(self, skip_c, in_c, out_c, up_stride, actfunc):
        super(DenseNet2D_up_block, self).__init__()
        self.conv11 = nn.Conv2d(skip_c+in_c, out_c, kernel_size=1, padding=0)
        self.conv12 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(skip_c+in_c+out_c, out_c, kernel_size=1,padding=0)
        self.conv22 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.actfunc = actfunc
        self.up_stride = up_stride

    def forward(self, prev_feature_map, x):
        x = F.interpolate(x,
                          mode='bilinear',
                          align_corners=False,
                          scale_factor=self.up_stride)
        x = torch.cat([x, prev_feature_map], dim=1)
        x1 = self.actfunc(self.conv12(self.conv11(x)))
        x21 = torch.cat([x, x1],dim=1)
        out = self.actfunc(self.conv22(self.conv21(x21)))
        return out

class DenseNet_encoder(nn.Module):
    def __init__(self, in_c=1, chz=32, actfunc=F.leaky_relu, growth=1.5, norm=nn.BatchNorm2d):
        super(DenseNet_encoder, self).__init__()
        sizes = getSizes(chz, growth)
        interSize = sizes['enc']['inter']
        opSize = sizes['enc']['op']
        ipSize = sizes['enc']['ip']

        self.head = convBlock(in_c=1,
                                inter_c=chz,
                                out_c=chz,
                                actfunc=actfunc)
        self.down_block1 = DenseNet2D_down_block(in_c=ipSize[0],
                                                 inter_c=interSize[0],
                                                 op_c=opSize[0],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block2 = DenseNet2D_down_block(in_c=ipSize[1],
                                                 inter_c=interSize[1],
                                                 op_c=opSize[1],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block3 = DenseNet2D_down_block(in_c=ipSize[2],
                                                 inter_c=interSize[2],
                                                 op_c=opSize[2],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.down_block4 = DenseNet2D_down_block(in_c=ipSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=2,
                                                 norm=norm,
                                                 actfunc=actfunc)
        self.bottleneck = DenseNet2D_down_block(in_c=opSize[3],
                                                 inter_c=interSize[3],
                                                 op_c=opSize[3],
                                                 down_size=0,
                                                 norm=norm,
                                                 actfunc=actfunc)
    def forward(self, x):
        x = self.head(x) # chz
        skip_1, x = self.down_block1(x) # chz
        skip_2, x = self.down_block2(x) # 2 chz
        skip_3, x = self.down_block3(x) # 4 chz
        skip_4, x = self.down_block4(x) # 8 chz
        _, x = self.bottleneck(x)
        return skip_4, skip_3, skip_2, skip_1, x

class DenseNet_decoder(nn.Module):
    def __init__(self, chz, out_c, growth, actfunc=F.leaky_relu, norm=nn.BatchNorm2d):
        super(DenseNet_decoder, self).__init__()
        sizes = getSizes(chz, growth)
        skipSize = sizes['dec']['skip']
        opSize = sizes['dec']['op']
        ipSize = sizes['dec']['ip']

        self.up_block4 = DenseNet2D_up_block(skipSize[0], ipSize[0], opSize[0], 2, actfunc)
        self.up_block3 = DenseNet2D_up_block(skipSize[1], ipSize[1], opSize[1], 2, actfunc)
        self.up_block2 = DenseNet2D_up_block(skipSize[2], ipSize[2], opSize[2], 2, actfunc)
        self.up_block1 = DenseNet2D_up_block(skipSize[3], ipSize[3], opSize[3], 2, actfunc)
        
        self.c3_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[0], out_channels=3)
        self.c2_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[1], out_channels=3)
        self.c1_11 = nn.Conv2d(kernel_size=1, in_channels=opSize[2], out_channels=3)

        self.final = convBlock(chz, chz, out_c, actfunc)

    def forward(self, skip4, skip3, skip2, skip1, x):
         x3 = self.up_block4(skip4, x)
         x2 = self.up_block3(skip3, x3)
         x1 = self.up_block2(skip2, x2)
         x = self.up_block1(skip1, x1)
         o = self.final(x)
         return o

class DenseNet2D(nn.Module):
    def __init__(self,
                 chz=32,
                 growth=1.2,
                 actfunc=F.leaky_relu,
                 norm=nn.InstanceNorm2d,
                 selfCorr=False,
                 disentangle=False):
        super(DenseNet2D, self).__init__()

        self.sizes = getSizes(chz, growth)
        self.toggle = True
        self.selfCorr = selfCorr
        self.disentangle = disentangle
        self.disentangle_alpha = 2

        self.enc = DenseNet_encoder(in_c=1, chz=chz, actfunc=actfunc, growth=growth, norm=norm)
        self.dec = DenseNet_decoder(chz=chz, out_c=2, actfunc=actfunc, growth=growth, norm=norm)
        # self.elReg = regressionModule(self.sizes)
        self._initialize_weights()

        self.interpolate = nn.Upsample(size=(480, 640), mode='bilinear')

    def setDatasetInfo(self, numSets=2):
        # Produces a 1 layered MLP which directly maps bottleneck to the DS ID
        self.numSets = numSets
        self.dsIdentify_lin = linStack(num_layers=2,
                                       in_dim=self.sizes['enc']['op'][-1],
                                       hidden_dim=64,
                                       out_dim=numSets,
                                       bias=True,
                                       actBool=False,
                                       dp=0.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):

        B, _, H, W = x.shape
        x4, x3, x2, x1, x = self.enc(x)
        latent = torch.mean(x.flatten(start_dim=2), -1) # [B, features]

        op = self.dec(x4, x3, x2, x1, x)

        op = self.interpolate(op)

        #return op, elOut, pul_exist 
        return op  

# class classifier(torch.nn.Module):
#     def __init__(self, sizes):
#         super(classifier, self).__init__()
#         inChannels = sizes['enc']['op'][-1] 
#         self.max_pool = nn.AvgPool2d(kernel_size=2)

#         self.c1 = nn.Conv2d(in_channels=inChannels,
#                             out_channels=64,
#                             bias=True,
#                             kernel_size=3,
#                             padding=1)

#         self.c2 = nn.Conv2d(in_channels=64,
#                             out_channels=64,
#                             bias=True,
#                             kernel_size=3,
#                             padding=1)

#         self.c3 = nn.Conv2d(in_channels=64+inChannels,
#                             out_channels=32,
#                             kernel_size=1,
#                             bias=True)

#         self.l1 = nn.Linear(32, 2, bias=True)

#         self.bn1 = nn.BatchNorm2d(num_features=inChannels)
#         self.bn2 = nn.BatchNorm2d(num_features=64+inChannels)

#     def forward(self, x):
#         B = x.shape[0]
#         p = self.bn1(x)
#         p = self.c1(p)
#         p = self.c2(p)
#         x = torch.cat([x, p], dim=1)
#         x = self.bn2(x)
#         x = self.c3(x)
#         x = self.l1(x.reshape(B, 32, -1).sum(dim=-1))
#         return x


'''
remove the regression module
'''
# class regressionModule(torch.nn.Module):
#     def __init__(self, sizes):
#         super(regressionModule, self).__init__()
#         inChannels = sizes['enc']['op'][-1]
#         self.max_pool = nn.AvgPool2d(kernel_size=2)

#         self.c1 = nn.Conv2d(in_channels=inChannels,
#                             out_channels=64,
#                             bias=True,
#                             kernel_size=3,
#                             padding=1)

#         self.c2 = nn.Conv2d(in_channels=64,
#                             out_channels=64,
#                             bias=True,
#                             kernel_size=3,
#                             padding=1)

#         self.c3 = nn.Conv2d(in_channels=64+inChannels,
#                             out_channels=32,
#                             kernel_size=1,
#                             bias=True)

#         self.l1 = nn.Linear(32, 128, bias=True)
#         self.l2 = nn.Linear(128, 5, bias=True)
#         self.bn1 = nn.BatchNorm2d(num_features=inChannels)
#         self.bn2 = nn.BatchNorm2d(num_features=64+inChannels)

#         self.xy_actfunc = F.hardtanh # Center has to be between -1 and 1
#         self.hw_actfunc = F.hardtanh # Parameters can't be negative and capped to 1
#         self.t_actfunc  = F.hardtanh # Parameters can't be negative and capped to 1

#     def forward(self, x):
#         # x: [B, C, 15, 20]
#         B = x.shape[0]
#         p = self.bn1(x)
#         p = self.c1(p)
#         p = self.c2(p)
#         x = torch.cat([x, p], dim=1)
#         x = self.bn2(x)
#         x = self.c3(x)
#         x = self.l1(x.reshape(B, 32, -1).sum(dim=-1))
#         x = self.l2(x)

#         EPS = 1e-5

#         pup_c     = self.xy_actfunc(x[:, 0:2], min_val=0+EPS, max_val=1-EPS)
#         pup_param = self.hw_actfunc(x[:, 2:4], min_val=0+EPS, max_val=1-EPS)
#         pup_angle = self.t_actfunc (x[:, 4], min_val=-1+EPS, max_val=1-EPS)

#         op = torch.cat([pup_c,
#                         pup_param,
#                         pup_angle.unsqueeze(1)], dim=1)

#         return op

if __name__ == '__main__':
    model = DenseNet2D()

    B = 3
    H = 192
    W = 256

    x = torch.rand(B, 1, H, W)

    #op, elOut, pul_exist = model.forward(x)
    op = model.forward(x)

    print(op.shape)