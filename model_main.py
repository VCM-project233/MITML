import copy

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet18
import torch.nn.functional as F
import math

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        t_x = self.layer4(x)
        x = self.base.layer4(x)
        return x,t_x

class TemporalMemory(nn.Module):
    def __init__(self, feat_dim=2048, mem_size=100, margin=1, seq_len=6):
        super(TemporalMemory, self).__init__()
        self.key = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.val = nn.Parameter(torch.empty(mem_size, seq_len).uniform_().cuda())
        self.lstm = nn.LSTM(feat_dim, feat_dim, 1)
        self.margin = margin
        self.S = seq_len

    def forward(self, query, val):
        query = query.reshape(query.shape[0]//self.S, self.S, -1).permute(1, 0, 2)
        h0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        c0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(query, (h0, c0))
        query_lstm = output[-1] # [B, F]

        similarity = torch.matmul(F.normalize(query_lstm, dim=1),
                                  F.normalize(self.key.t(), dim=1))
        r_att = F.softmax(similarity, dim=1)
        read = F.softmax(torch.matmul(r_att, self.val), dim=1)

        val = val.reshape(val.shape[0]//self.S, self.S, -1)
        out = torch.bmm(read.unsqueeze(1), val).squeeze(1)
        return {'out':out, 'loss':self.loss(r_att, self.margin)}

    def loss(self, r_att, margin=1):
        topk = r_att.topk(r_att.shape[0], dim=0)[0]
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return {'mem_trip':mem_trip}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.l = nn.Conv2d(channel, channel,1)
    def forward(self, x,f):
        y = self.fc(x)
        return f* y+f

def conv1x1(conv,x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x

class temporal_feat_learning(nn.Module):
    def __init__(self,  ):
        super(temporal_feat_learning, self).__init__()
        dim = 2048
        self.se_1 = SELayer(2048)
        self.se_2 = SELayer(2048)
        self.se_3 = SELayer(2048)
        self.se_4 = SELayer(2048)
        self.se_5 = SELayer(2048)
        self.se_6 = SELayer(2048)

        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.f = nn.Linear(dim, dim)


    def forward(self, t_x,x,x_h):
        t1 = self.a(t_x)+x[0]
        t2 = self.b(t_x)+x[1]
        t3 = self.c(t_x)+x[2]
        t4 = self.d(t_x)+x[3]
        t5 = self.e(t_x)+x[4]
        t6 = self.f(t_x)+x[5]

        f1 = self.se_1(t1/2,x_h[0]).unsqueeze(dim=1)
        f2 = self.se_2(t2/2,x_h[1]).unsqueeze(dim=1)
        f3 = self.se_3(t3/2,x_h[2]).unsqueeze(dim=1)
        f4 = self.se_4(t4/2,x_h[3]).unsqueeze(dim=1)
        f5 = self.se_5(t5/2,x_h[4]).unsqueeze(dim=1)
        f6 = self.se_6(t6/2,x_h[5]).unsqueeze(dim=1)

        f = torch.cat((f1,f2,f3,f4,f5,f6),dim=1)
        f = f.mean(dim=1)

        return f


class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]

class embed_net(nn.Module):
    def __init__(self,  low_dim,  class_num, drop=0.2, part = 3, alpha=0.2, nheads=4, arch='resnet50', wpa = False):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)


        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.bottleneck1.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier1.apply(weights_init_classifier)
        self.classifier2.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 2048, 2)
        self.temporal_feat_learning = temporal_feat_learning()

    def forward(self, x1, x2, adj, modal=0, seq_len = 8, cpa = False):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x,x_t = self.base_resnet(x)
        x_l = self.avgpool(x_t).squeeze()
        x_l = x_l.view(x_l.size(0)//t, t, -1).permute(1, 0, 2)

        x_h = self.avgpool(x).squeeze()
        x_h = x_h.view(x_h.size(0)//t, t, -1).permute(1, 0, 2)

        h0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()
        c0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x_l, (h0, c0))
        t = output[-1]
        x_pool = self.temporal_feat_learning(t,x_l,x_h)
        feat  = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(feat)