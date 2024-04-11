from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module import FACMA

logger = logging.getLogger(__name__)
class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
			stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rates):
        super(ASPP, self).__init__()

        self.aspp1 = ASPP_module(inplanes, planes, rate=rates[0])
        self.aspp2 = ASPP_module(inplanes, planes, rate=rates[1])
        self.aspp3 = ASPP_module(inplanes, planes, rate=rates[2])
        self.aspp4 = ASPP_module(inplanes, planes, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.ReLU()
		)
        self.conv1 = nn.Conv2d(planes*5, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class WCMF(nn.Module):
    def __init__(self,channel=256):
        super(WCMF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_d1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

        self.conv_c1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def fusion(self,f1,f2,f_vec):

        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2
    def forward(self,rgb,depth):
        Fr = self.conv_r1(rgb)
        Fd = self.conv_d1(depth)
        f = torch.cat([Fr, Fd],dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        # f = self.avgpool(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        fidx_u = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
        fidx_v = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]

        self.FACMA1 = FACMA(128, 64, 64, fidx_u, fidx_v)
        self.FACMA2 = FACMA(256, 32, 32, fidx_u, fidx_v)
        self.FACMA3 = FACMA(512, 16, 16, fidx_u, fidx_v)
        self.FACMA4 = FACMA(512,  8,  8, fidx_u, fidx_v)


        self.WCMF2 = WCMF(128)
        self.WCMF3 = WCMF(256)
        self.WCMF4 = WCMF(512)
        self.WCMF5 = WCMF(512)

        self.relu = nn.ReLU(inplace=True)
        self.cp2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                   nn.Conv2d(128, 64, 3, 1, 1), self.relu)

        self.cp3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                   nn.Conv2d(128, 64, 3, 1, 1), self.relu)

        self.cp4 = nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 128, 5, 1, 2), self.relu,
                                   nn.Conv2d(128, 64, 3, 1, 1), self.relu)

        self.cp5 = nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 128, 5, 1, 2), self.relu,
                                   nn.Conv2d(128, 64, 3, 1, 1), self.relu)

        rates = [1, 6, 12, 18]
        self.ASPP1 = ASPP(64, 64, rates)
        self.ASPP2 = ASPP(64, 64, rates)
        self.ASPP3 = ASPP(64, 64, rates)
        self.ASPP4 = ASPP(64, 64, rates)


        self.conv_2 = nn.Conv2d(64, 1, 3, 1, 1)

        self.conv_3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_4 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_5 = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_o = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, img, h2, h3, h4, h5, d2, d3, d4, d5):
        raw_size = img.size()[2:]

        rf2, rd2 = self.FACMA1(h2, d2)     # 64*64*128
        rf3, rd3 = self.FACMA2(h3, d3)     # 32*32*256
        rf4, rd4 = self.FACMA3(h4, d4)     # 16*16*512
        rf5, rd5 = self.FACMA4(h5, d5)     # 8 *8 *512

        F2 = self.WCMF2(rf2, rd2)
        F3 = self.WCMF3(rf3, rd3)
        F4 = self.WCMF4(rf4, rd4)
        F5 = self.WCMF5(rf5, rd5)

        F2 = self.cp2(F2)
        F3 = self.cp3(F3)
        F4 = self.cp4(F4)
        F5 = self.cp5(F5)
        # print("//////", F5.shape)
        F5_A = self.ASPP1(F5)
        F4_A = self.ASPP2(F4 + F.interpolate(F5_A, F4.shape[2:], mode='bilinear'))
        F3_A = self.ASPP3(F3 + F.interpolate(F5_A, F3.shape[2:], mode='bilinear') + F.interpolate(F4_A, F3.shape[2:],
                                                                                                  mode='bilinear'))
        F2_A = self.ASPP4(F2 + F.interpolate(F5_A, F2.shape[2:], mode='bilinear') + F.interpolate(F4_A, F2.shape[2:],
                                             mode='bilinear') + F.interpolate(F3_A, F2.shape[2:], mode='bilinear'))

        Fo_2 = F.interpolate(self.conv_2(F2), raw_size, mode='bilinear')
        Fo_3 = F.interpolate(self.conv_3(F3), raw_size, mode='bilinear')
        Fo_4 = F.interpolate(self.conv_4(F4), raw_size, mode='bilinear')
        Fo_5 = F.interpolate(self.conv_5(F5), raw_size, mode='bilinear')

        Fo = F.interpolate(self.conv_2(F2_A), raw_size, mode='bilinear')
        return Fo_2, Fo_3, Fo_4, Fo_5, Fo
    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


