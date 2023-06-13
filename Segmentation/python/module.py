#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import mobilenet_v3_large, resnet50
# from torchvision.models.segmentation import deeplabv3_resnet50
# from module_deep import *
# from module_deep import DeepLabHead
# from utils_deep import *
# from backbone_mobile import *


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

        return op  
#################-------------------------------------------##########################
class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()

        self.model = mobilenet_v3_large(pretrained = True) 
        list(self.model.features.children())[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        list(self.model.features.children())[0][1] = torch.nn.BatchNorm2d(16) 
        #del list(self.model.features.children())[1]
        #self.model.classifier = nn.Sequential(self.model.classifier[0], 
        #                                    nn.Linear(in_features=1280, out_features=100, bias=True),
        #                                    nn.Linear(in_features=100, out_features=2, bias=True)) 
        #self.interpolate = nn.Upsample(size=(480, 640), mode='bilinear')
    def forward(self, x):
        # print(x.size)
        x = self.model(x)
        x = self.interpolate(x)
        return x

################---------------------------------------#######################################
class deeplabv3(nn.Module):
    def __init__(self):
        super(deeplabv3, self).__init__()

        self.model = deeplabv3_resnet50(pretrained=True) 
        
        self.model.backbone.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.backbone.bn1 = torch.nn.BatchNorm2d(16)
        self.model.backbone.layer1[0].conv1 = nn.Conv2d(16, 64, kernel_size=(3, 3))
        #list(self.model.features.children())[0][1] = torch.nn.BatchNorm2d(16) 
        #del list(self.model.features.children())[1]
#
        self.model.aux_classifier = nn.Sequential(self.model.classifier[0], 
                                            nn.Linear(in_features=1280, out_features=100, bias=True),
                                            nn.Linear(in_features=100, out_features=2, bias=True)) 
        #self.interpolate = nn.Upsample(size=(480, 640), mode='bilinear')
    def forward(self, x):
        x = self.model(x)
        return x
################----------------------------###########################################
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = Identity()

        self.conf_fc = nn.Linear(2048, 2) 
        self.interpolate = nn.Upsample(size=(480, 640), mode='bilinear')
        
    def forward(self, x):
        x = self.resnet(x)
        return  x

########---------------------------------------------------##############

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model
#########--------------------------------------------######################
def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 1
    low_level_planes = 24
    
    #if name=='deeplabv3plus':
    #    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    #    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    if name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate) ###222222DDDD
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return backbone.low_level_features



if __name__ == '__main__':
    #model = DenseNet2D()
    model = _segm_mobilenet('deeplabv3', 'mobilenetv2', 2, 2, True)
    #model = deeplabv3()
    #model = ResNet50()
    print(model)
    B = 3
    H = 192
    W = 256

    x = torch.rand(B, 1, H, W)
    op = model.forward(x)

    print(op.shape)