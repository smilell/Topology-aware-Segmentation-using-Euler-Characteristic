import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class UNet2D(nn.Module):

    def __init__(self, num_classes, ):
        super(UNet2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv22 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv13 = nn.Conv2d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv2d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))

        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)
        return x


class UNet3D(nn.Module):

    def __init__(self, num_classes):
        super(UNet3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(1, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv12 = nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=use_bias)
        self.down1 = nn.Conv3d(8, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        # odd e.g. in_size 11 -> out_size = upper(11/2) -> 6
        # even e.g., in_size=12 -> out_size = 12/2 -> 6

        self.conv21 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)

        self.conv31 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2, bias=use_bias)

        self.conv41 = nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=use_bias)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(96, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = nn.Conv3d(24, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv14 = nn.Conv3d(8, num_classes, kernel_size=1, padding=0, bias=use_bias)


    def forward(self, x):
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        x2 = self.down1(x1)
        x2 = F.relu(self.conv21(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        self.neck = x4
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv22(x2))
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = F.relu(self.conv13(x1))
        x = self.conv14(x1)

        return x

    def get_neck(self):
        return self.neck


    # x_filter = np.array([[1, 0, -1],
    #                      [2, 0, -2],
    #                      [1, 0, -1]]).reshape(1, 1, 3, 3)
    #
    # x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    # x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    # conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
    #                    dilation=1, bias=False)
    #
    # conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())
    #
    # y_filter = np.array([[1, 2, 1],
    #                      [0, 0, 0],
    #                      [-1, -2, -1]]).reshape(1, 1, 3, 3)
    # y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
    # y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
    # conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
    #                    bias=False)
    # conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())
    #
    # conv_x = conv_x.to(input.device)
    # conv_y = conv_y.to(input.device)
    # for param in conv_y.parameters():
    #     param.requires_grad = False
    # for param in conv_x.parameters():
    #     param.requires_grad = False


class EC(nn.Module):
    def __init__(self, subpatch_size=32, subpatch_stride=8):
        super(EC, self).__init__()

        self.subpatch_size = subpatch_size
        self.overlap_size = subpatch_stride
        self.isApproxBinary = True

        self.CeConv_1 = nn.Conv2d(1, 4, kernel_size=2, bias=False)
        self.CeConv_2 = nn.Conv2d(1, 4, kernel_size=2, bias=False)
        self.CeConv_3 = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        # [4, 2, 2], [4, 2, 2], [2, 2, 2]
        x_filter_1 = np.array([[[1, -1], [-1, -1]],
                             [[-1, 1], [-1, -1]],
                             [[-1, -1], [-1, 1]],
                             [[-1, -1], [1, -1]],]
                             ).reshape(4, 1, 2, 2)

        x_filter_2 = np.array([[[1, 1], [1, -1]],
                               [[1, 1], [-1, 1]],
                               [[-1, 1], [1, 1]],
                               [[1, -1], [1, 1]]
                               ]).reshape(4, 1, 2, 2)

        x_filter_3 = np.array([[[1, -1], [-1, 1]],
                               [[-1, 1], [1, -1]],
                               ]).reshape(2, 1, 2, 2)


        self.CeConv_1.weight = nn.Parameter(torch.from_numpy(x_filter_1).float())
        self.CeConv_2.weight = nn.Parameter(torch.from_numpy(x_filter_2).float())
        self.CeConv_3.weight = nn.Parameter(torch.from_numpy(x_filter_3).float())

        for param in self.CeConv_1.parameters():
            param.requires_grad = False
        for param in self.CeConv_2.parameters():
            param.requires_grad = False
        for param in self.CeConv_3.parameters():
            param.requires_grad = False#False

        self.AvePool = nn.AvgPool2d(self.subpatch_size, stride=self.overlap_size)


    def forward(self, x_hard, pad_value=-1, isReturnEcMap=True, isReturnInternalFeatures=False):
        return_dist = {}
        p2d = (1,1,1,1)

        '''EC hard'''
        x_norm_hard = x_hard * 2 -1

        x_norm_pad_hard = nn.functional.pad(x_norm_hard, p2d, 'constant', pad_value)
        '''1.1 EC_map unnormalized'''
        f1_hard_norm = self.CeConv_1(x_norm_pad_hard)
        f2_hard_norm = self.CeConv_2(x_norm_pad_hard)       # if x is binary, feature 2 should either be 0, 1/3, 2/3 or 1, and only count the number of 1
        f3_hard_norm = self.CeConv_3(x_norm_pad_hard)
        '''1.2 EC_map_normalized'''
        f1_hard_norm[f1_hard_norm <= 3.9] = 0
        f2_hard_norm[f2_hard_norm <= 3.9] = 0
        f3_hard_norm[f3_hard_norm <= 3.9] = 0
        f1_hard_norm = f1_hard_norm/4
        f2_hard_norm = f2_hard_norm / 4
        f3_hard_norm = f3_hard_norm / 4

        '''1.3 EC_NUMBER'''
        EC_hard_1 = f1_hard_norm.sum(dim=(1, 2, 3))
        EC_hard_2 = f2_hard_norm.sum(dim=(1, 2, 3))
        EC_hard_3 = f3_hard_norm.sum(dim=(1, 2, 3))
        EC_hard = 1/4 * (EC_hard_1 - EC_hard_2 - 2 * EC_hard_3)

        return_dist.update({'EC': EC_hard})

        if isReturnInternalFeatures:
            return_dist.update({'EC_features': [f1_hard_norm.sum(dim=1), f2_hard_norm.sum(dim=1), f3_hard_norm.sum(dim=1)]})

        '''1.4 EC numbers on subpatch'''
        if isReturnEcMap:
            EC_hard_patch_1 = self.subpatch_size * self.subpatch_size * self.AvePool(f1_hard_norm.sum(dim=1))
            EC_hard_patch_2 = self.subpatch_size * self.subpatch_size * self.AvePool(f2_hard_norm.sum(dim=1))
            EC_hard_patch_3 = self.subpatch_size * self.subpatch_size * self.AvePool(f3_hard_norm.sum(dim=1))

            EC_hard_patch = 1/4 * (EC_hard_patch_1 - EC_hard_patch_2 - 2 * EC_hard_patch_3)
            return_dist.update({'EC_map': EC_hard_patch})

        return return_dist


