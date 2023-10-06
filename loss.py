import math
from collections import OrderedDict
import time
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import gudhi as gd
import cripser
import cv2
import os
import SimpleITK as sitk
from scipy import ndimage
import multiprocessing as mlp
# import ripserplusplus as rpp_py
import cc3d
# from utilities import RobustCrossEntropyLoss, SoftDiceLoss, softmax_helper, DC_and_CE_loss



class LossFn(nn.Module):
    def __init__(self,
                 sim_loss_fn,
                 reg_loss_fn,
                 sim_loss_weight=1.,
                 reg_loss_weight=1.
                 ):
        super(LossFn, self).__init__()
        self.sim_loss_fn = sim_loss_fn
        self.sim_loss_weight=sim_loss_weight
        self.reg_loss_fn = reg_loss_fn
        self.reg_loss_weight = reg_loss_weight

    def forward(self, tar, warped_src, u):
        sim_loss = self.sim_loss_fn(tar, warped_src)
        reg_loss = self.reg_loss_fn(u)
        loss = sim_loss * self.sim_loss_weight + reg_loss * self.reg_loss_weight
        return {'sim_loss': sim_loss,
                'reg_loss': reg_loss,
                'loss': loss}


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# def active_contour_loss(y_true, y_pred, weight=10):
#     '''
#     y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
#     weight: scalar, length term weight.
#     '''
#     # length term
#     delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
#     delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)
#
#     delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
#     delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
#     delta_pred = torch.abs(delta_r + delta_c)
#
#     epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
#     lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.
#
#     # region term
#     c_in = torch.ones_like(y_pred)
#     c_out = torch.zeros_like(y_pred)
#
#     region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
#     region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
#     region = region_in + region_out
#
#     loss = weight * lenth + region
#
#     return loss


def contour_loss(input, target, size_average=True, use_gpu=True, ignore_background=False, one_hot_target=True, mask=None, return_map=False):
    '''
    calc the contour loss across object boundaries (WITHOUT background class)
    :param input: NDArray. N*num_classes*H*W : pixelwise probs. for each class e.g. the softmax output from a neural network
    :param target: ground truth labels (NHW) or one-hot ground truth maps N*C*H*W
    :param size_average: batch mean
    :param use_gpu:boolean. default: True, use GPU.
    :param ignore_background:boolean, ignore the background class. default: True
    :param one_hot_target: boolean. if true, will first convert the target from NHW to NCHW. Default: True.
    :return:
    '''
    n, num_classes, h, w = input.size(0), input.size(
        1), input.size(2), input.size(3)
    # if one_hot_target:
    #     onehot_mapper = One_Hot(depth=num_classes, use_gpu=use_gpu)
    #     target = target.long()
    #     onehot_target = onehot_mapper(target).contiguous().view(
    #         input.size(0), num_classes, input.size(2), input.size(3))
    # else:
    onehot_target = target
    assert onehot_target.size() == input.size(), 'pred size: {} must match target size: {}'.format(
        str(input.size()), str(onehot_target.size()))

    if mask is None:
        # apply masks so that only gradients on certain regions will be backpropagated.
        mask = torch.ones_like(input).long().to(input.device)
        mask.requires_grad = False
    else:
        pass
        # print ('mask applied')

    if ignore_background:
        object_classes = num_classes - 1
        target_object_maps = onehot_target[:, 1:].float()
        input = input[:, 1:]
    else:
        target_object_maps = onehot_target
        object_classes = num_classes

    x_filter = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]]).reshape(1, 1, 3, 3)

    x_filter = np.repeat(x_filter, axis=1, repeats=object_classes)
    x_filter = np.repeat(x_filter, axis=0, repeats=object_classes)
    conv_x = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       dilation=1, bias=False)

    conv_x.weight = nn.Parameter(torch.from_numpy(x_filter).float())

    y_filter = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]).reshape(1, 1, 3, 3)
    y_filter = np.repeat(y_filter, axis=1, repeats=object_classes)
    y_filter = np.repeat(y_filter, axis=0, repeats=object_classes)
    conv_y = nn.Conv2d(in_channels=object_classes, out_channels=object_classes, kernel_size=3, stride=1, padding=1,
                       bias=False)
    conv_y.weight = nn.Parameter(torch.from_numpy(y_filter).float())

    conv_x = conv_x.to(input.device)
    conv_y = conv_y.to(input.device)
    for param in conv_y.parameters():
        param.requires_grad = False
    for param in conv_x.parameters():
        param.requires_grad = False

    g_x_pred = conv_x(input) * mask[:, :object_classes]
    g_y_pred = conv_y(input) * mask[:, :object_classes]
    g_y_truth = conv_y(target_object_maps) * mask[:, :object_classes]
    g_x_truth = conv_x(target_object_maps) * mask[:, :object_classes]

    # mse loss
    loss = torch.nn.MSELoss(reduction='mean')(input=g_x_pred, target=g_x_truth) + \
        torch.nn.MSELoss(reduction='mean')(input=g_y_pred, target=g_y_truth)
    loss = 0.5 * loss
    if return_map:
        return loss, g_x_pred, g_y_pred, g_x_truth, g_y_truth
    return loss


class weightedDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(weightedDiceLoss, self).__init__()

    def forward(self, inputs, targets, weight, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        weight = weight.view(-1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class softDiceLoss(nn.Module):
    def __init__(self):
        super(softDiceLoss, self).__init__()

    def one_class_dice(self, label, pred, smooth = 1):
        label = label.view(-1)
        pred = pred.view(-1)

        inter = (label * pred).sum()
        dice = (2 * inter + smooth) / (label.sum() + pred.sum() + smooth)
        return dice

    def forward(self, label, pred, num_classes):
        ''' both label and pred are one-hot in [batch, num_classes, x,y,z]'''
        for i in range(num_classes):

            class_label = label[:, i, :, :, :]
            class_pred = pred[:, i, :, :, :]
            if i ==0:
                dice = self.one_class_dice(class_label, class_pred)
            else:
                dice += self.one_class_dice(class_label, class_pred)

        return 1- torch.mean(dice)/num_classes


class topo_loss(nn.Module):
    def __init__(self, package='cripser'):
        super(topo_loss, self).__init__()
        '''package = gudhi (8-connectivity in 2d) or cripser (4-connectivity in 2d) '''
        self.package = package

    def compute_dgm_force_new(self, lh_dgm, gt_dgm, pers_thresh=0, pers_thresh_perfect=0.99, do_return_perfect=False):
        idx_fix_holes = {}
        idx_remove_holes = {}

        for dim in list(lh_dgm.keys()):
            idx_fix_holes.update({dim: []})
            idx_remove_holes.update({dim: []})
            dim_int = int(dim)
            lh_pers = abs(lh_dgm[dim][:, 1] - lh_dgm[dim][:, 0])
            lh_pers_idx_ranked = np.argsort(lh_pers)[::-1]
            lh_n = len(lh_pers)

            if dim in gt_dgm.keys():
                gt_pers = abs(gt_dgm[dim][:, 1] - gt_dgm[dim][:, 0])
                gt_n = len(gt_pers)
                assert np.array_equal(gt_pers, np.ones(gt_n))
            else:
                gt_pers = None
                gt_n = 0

            '''the number of likelihood complex > gt: some of them fixed and some of them removed '''
            if lh_n > gt_n:
                N_holes_2_fix = gt_n
                N_holes_2_remove = lh_n - gt_n

                idx_fix_holes.update({dim: lh_pers_idx_ranked[0:gt_n]})
                idx_remove_holes.update({dim: lh_pers_idx_ranked[gt_n::]})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix
                assert len(idx_remove_holes[dim]) == N_holes_2_remove
            elif lh_n <= gt_n:
                N_holes_2_fix = lh_n
                N_holes_2_remove = 0
                idx_fix_holes.update({dim: lh_pers_idx_ranked})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix

        return idx_fix_holes, idx_remove_holes

    # def recover(self, dim_eff):

    def pre_process(self, info):
        '''info in shape [dim, b, d, b_x, b_y. b_z, d_x, d_y, d_z]'''


        revised_row_array = np.where(info[:, 2]>1)[0]
        for row in revised_row_array:
            info[row, 2] = 1

        dim_eff_all = np.unique(info[:,0])
        pd_gt_1 = {}
        bcp_gt_1 = {}
        dcp_gt_1 = {}

        for dim_eff in dim_eff_all:
            idx = info[:, 0] == dim_eff
            pd_gt_1.update({str(int(dim_eff)): info[idx][:, 1:3]})
            bcp_gt_1.update({str(int(dim_eff)): info[idx][:, 3:6]})
            dcp_gt_1.update({str(int(dim_eff)): info[idx][:, 6::]})

        return pd_gt_1, bcp_gt_1, dcp_gt_1

    def check_point_exist(self, r_max, p):
        result = True
        for i, point in enumerate(list(p)):
            if point < 0 or point >= r_max[i]:
                result = False
        return result

    def get_prediction_map(self, pd_lh, map, name):
        root = '/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_5/betti_new/'
        threshold_born = 1 - pd_lh[0]
        threshold_death = 1 - pd_lh[1]

        map_b= torch.zeros(map.shape)
        map_d = torch.zeros(map.shape)
        map_b[map > threshold_born] = 1
        map_d[map > threshold_death] = 1
        self.save_nii([root + x + name + '_p_' + str(threshold_born) + '.nii.gz' for x in ['map_born_', ]], map_b)
        self.save_nii([root + x + name + '_p_' + str(threshold_death) + '.nii.gz' for x in ['map_death_']], map_d)
        
    def save_B_D_points(self, map, pd_lh, bcp_lh, dcp_lh, no =1000):
        result = np.zeros(map.shape)
        result_inv = np.zeros(map.shape)
        for i in range(no):
            coor_b = tuple(bcp_lh['0'][i].astype(np.int))
            coor_d = tuple(dcp_lh['0'][i].astype(np.int))
            # coor_b_inv = tuple(bcp_lh['0'][i].astype(np.int)[::-1])
            # coor_d_inv = tuple(dcp_lh['0'][i].astype(np.int)[::-1])
            result[coor_b] = i
            result[coor_d] = -1 *i
            
            # result_inv[coor_b_inv] = i
            # result_inv[coor_d_inv] = -1 *i
        self.save_nii(['/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/points.nii.gz'],
                       #'/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/points_inv.nii.gz'],
                      result, )
                      #result_inv)
    '''old version'''
    # def reidx_f_gudhi(self, idx, ori_shape):
    #     ''' given the original shape of a map [x_ori, y_ori, z_ori], and an index in the range of np.prod(ori_shape), find the coordinate index in original image'''
    #     ''' try two options:
    #         (1) first occupy ori_shape[0] then goes to ori_shape[1]...
    #         e.g., ori_shape = [2, 3, 4] and idx_max = 2*3*4-1 = 23, coordinate_max= [1, 2, 3]
    #         idx= 4, then return coordinate = [0, 2, 0]
    #         (2) first occupy ori_shape[-1] then goes to ori_shape[-1-1]...
    #         e.g., return coordinate = [0, 1, 0]
    #     '''
    #     re_idx = np.zeros(3, dtype=np.uint8)
    #     reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
    #     reidx_1 = np.array(np.unravel_index(idx, ori_shape, order='F'))
    #     if len(ori_shape) == 3:
    #         # div_0 = ori_shape[1] * ori_shape[2]
    #         # re_idx[0] = int(idx // div_0)
    #         # mod_0 = idx % div_0
    #         #
    #         # re_idx[1] = int(mod_0 // ori_shape[1])
    #         # re_idx[2] = int(mod_0 % ori_shape[1])
    #         # assert idx == re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[1] + re_idx[2]
    #
    #         div_0 = ori_shape[1] * ori_shape[0]
    #         re_idx[2] = int(idx // div_0)
    #         mod_0 = idx % div_0
    #
    #         re_idx[1] = int(mod_0 // ori_shape[0])
    #         re_idx[0] = int(mod_0 % ori_shape[0])
    #         assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
    #         assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
    #
    #         if not (re_idx[0]<=ori_shape[0] and re_idx[1]<=ori_shape[1] and re_idx[2]<=ori_shape[2]):
    #             print('hi')
    #
    #     elif len(ori_shape) == 2:
    #         re_idx[0] = int(idx // ori_shape[1])
    #         re_idx[1] = int(idx % ori_shape[1])
    #         assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
    #         assert (reidx_1 == re_idx).all(), 'the reindex is not following mode C, unless the input image size is square'
    #
    #         # re_idx[0] = int(idx // ori_shape[0])
    #         # re_idx[1] = int(idx % ori_shape[0])
    #         # assert idx == re_idx[0] * ori_shape[0] + re_idx[1]
    #
    #     return re_idx
    def reidx_f_gudhi(self, idx, ori_shape):
        re_idx = np.zeros(3, dtype=np.uint16)
        reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
        reidx_1 = np.array(np.unravel_index(idx, ori_shape, order='F'))
        if len(ori_shape) == 3:
            div_0 = ori_shape[1] * ori_shape[2]
            re_idx[0] = int(idx // div_0)
            mod_0 = idx % div_0

            re_idx[1] = int(mod_0 // ori_shape[2])  # updated on 23/09/02 from ori_shape[1] to ori_shape[2]
            re_idx[2] = int(mod_0 % ori_shape[2])
            if idx != re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[2] + re_idx[
                2]:  # updated on 23/09/02 from re_idx[1] * ori_shape[1] to re_idx[1] * ori_shape[2]
                print('hold on, wrong reidx')
            assert (
                        reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

            # div_0 = ori_shape[1] * ori_shape[0]
            # re_idx[2] = int(idx // div_0)
            # mod_0 = idx % div_0
            #
            # re_idx[1] = int(mod_0 // ori_shape[0])
            # re_idx[0] = int(mod_0 % ori_shape[0])
            # assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
            # assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
            #
            # if not (re_idx[0] <= ori_shape[0] and re_idx[1] <= ori_shape[1] and re_idx[2] <= ori_shape[2]):
            #     print('hi')

        elif len(ori_shape) == 2:
            re_idx[0] = int(idx // ori_shape[1])
            re_idx[1] = int(idx % ori_shape[1])
            assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
            assert (
                        reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

            # re_idx[1] = int(idx // ori_shape[0])
            # re_idx[0] = int(idx % ori_shape[0])
            # assert idx == re_idx[1] * ori_shape[0] + re_idx[0]
        return re_idx  # re_idx

    def get_info_gudhi(self, map):
        cc = gd.CubicalComplex(dimensions=map.shape[::-1], top_dimensional_cells=1 - map.flatten())
        ph = cc.persistence()
        # betti_2 = cc.persistent_betti_numbers(from_value=1, to_value=0)
        x = cc.cofaces_of_persistence_pairs()

        '''3.1 get birth and death point coordinate from gudhi, and generate info array'''
        info_gudhi = np.zeros((len(ph), 9))
        # x will lack one death point where the filtration is inf
        '''3.1.1 manually write the inf death point'''
        reidx_birth_0 = self.reidx_f_gudhi(x[1][0][0], map.shape)
        if len(map.shape) == 2:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1]]
        elif len(map.shape) == 3:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2]]

        info_gudhi[0, :] = [0, birth_filtration, 1,
                            reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2],
                            0, 0, 0]
        idx_row = 1
        for dim in range(len(x[0])):
            for idx in range(x[0][dim].shape[0]):
                idx_brith, idx_death = x[0][dim][idx]
                reidx_birth = self.reidx_f_gudhi(idx_brith, map.shape)
                reidx_death = self.reidx_f_gudhi(idx_death, map.shape)

                if len(map.shape) == 2:
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1]]
                elif len(map.shape) == 3:
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1], reidx_birth[2]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1], reidx_death[2]]
                else:
                    assert False, 'wrong input dimension!'

                info_gudhi[idx_row, :] = [dim, birth_filtration, death_filtration,
                                          reidx_birth[0], reidx_birth[1], reidx_birth[2],
                                          reidx_death[0], reidx_death[1], reidx_death[2]]
                idx_row += 1

        return info_gudhi


    def get_topo_loss(self, map, batch, cgm_idx):
        # info_gt = cripser.computePH(map, maxdim=3)  # dim, b, d, b_x, b_y. b_z
        # dims_gt, pd_gt, bcp_gt, dcp_gt = self.pre_process(info_gt)

        # ''' test_1 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map, maxdim=3)
        # t2 = time.time() - t1
        # print('time1: ', t2)
        #
        # map_discrete = (map * 10).astype(np.int)/10
        # ''' test_2 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map_discrete, maxdim=3)
        # t2 = time.time() - t1
        # print('time2: ', t2)
        #
        # map_discrete = (map * 100).astype(np.int)/100
        # ''' test_2 '''
        # t1 = time.time()
        # for i in range(100):
        #     cripser.computePH(1 - map_discrete, maxdim=3)
        # t2 = time.time() - t1
        # print('time3: ', t2)
        t1 =time.time()
        betti = np.zeros((1,3))
        if self.package == 'cripser':
            info_lh = cripser.computePH(1 - map, maxdim=3)
        elif self.package == 'gudhi':
            info_lh = self.get_info_gudhi(map)
        else:
            assert False, print('wrong package to calculate topology')
        t2 =time.time()
        # print('1. calculatePH: ', t2-t1)

        # map_discrete = (map * 10).astype(np.int)/10
        # info_lh_discrete = cripser.computePH(1 - map_discrete, maxdim=3)

        # for i in range(10, -1, -1):
        #     map_threshold = np.zeros(map.shape)
        #     thre = i/10
        #     map_threshold[map >= thre] =1
        #     self.save_nii(['output/itn_8/threshold/' + str(thre) + '.nii.gz'], map_threshold)

        pd_lh, bcp_lh, dcp_lh = self.pre_process(info_lh)
        for i in range(3):
            betti[0, i] = len(pd_lh[str(i)]) if str(i) in list(pd_lh.keys()) else 0

        '''check the topology with the top ranked ph length, only in dim 0'''
        deta = pd_lh['0'][:, 1] - pd_lh['0'][:, 0]
        rank = np.argsort(deta)
        # for i in range(5):
        #     self.get_prediction_map(pd_lh['0'][rank[-1 * i]], map, 'rank_' + str(i))
        
        '''show top 10 coordinate points that has the minimal born and death (prob ->1)'''
        # self.save_B_D_points(map, pd_lh, bcp_lh, dcp_lh)
        # self.save_nii( ['/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/dhcp_1122/betti_gudhi/ori.nii.gz'], map)
        # no_gt = int(0.05* len(pd_lh['0'][:, 1] )) if int(0.05* len(pd_lh['0'][:, 1] ))>1 else 1
        no_gt = 1
        a = np.zeros([no_gt, 2])
        a[:,1] = 1
        pd_gt = {'0': a}     #{'0': a, '2':a}        # np.array([[0, 1]])

        idx_holes_to_fix, idx_holes_to_remove = self.compute_dgm_force_new(pd_lh, pd_gt, pers_thresh=0)
        '''
        topo_cp value map:
        0： background
        1, 2, 3：point to fix (born: force it to 0)
        4, 5, 6: point to fix (death: force it to 1)
        7, 8, 9: point to remove (born: force it to the prob of death)
        10,11,12: point to remove (death: force it to the prob of born)
        '''
        topo_size = list(map.shape)
        topo_cp_weight_map = np.zeros(map.shape)
        topo_cp_ref_map = np.zeros(map.shape)

        for dim in ['0', '1', '2']:
            dim_int = int(dim)
            if dim in list(idx_holes_to_fix.keys()):
                for hole_indx in idx_holes_to_fix[dim]:
                    if self.check_point_exist(topo_size, bcp_lh[dim][hole_indx]):
                        # if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_bb = (coor_b[0], coor_b[1], coor_b[2])
                        topo_cp_weight_map[coor_bb] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_bb] = 0

                        # self.get_prediction_map(pd_lh[dim][hole_indx], map)

                        # topo_cp[coor_bb] = 1 + dim_int

                    # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                    if self.check_point_exist(topo_size, dcp_lh[dim][hole_indx]):
                        # if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_dd = (coor_d[0], coor_d[1], coor_d[2])
                        # topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        # topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                        topo_cp_weight_map[coor_dd] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_dd] = 1

                        # topo_cp[coor_dd] = 4 + dim_int
            no_1 = 0
            no_2 = 0
            np_3 = 0
            if dim in list(idx_holes_to_remove.keys()):
                for hole_indx in idx_holes_to_remove[dim]:
                    coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_bb = (coor_b[0], coor_b[1], coor_b[2])

                    coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_dd = (coor_d[0], coor_d[1], coor_d[2])

                    b_exists = self.check_point_exist(topo_size, bcp_lh[dim][hole_indx])
                    d_exists = self.check_point_exist(topo_size, dcp_lh[dim][hole_indx])
                    if b_exists and d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_bb] = map[coor_dd]
                        topo_cp_ref_map[coor_dd] = map[coor_bb]
                        # topo_cp[coor_bb] = 7 + dim_int
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_1 = no_1 + 1
                    elif b_exists and not d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_ref_map[coor_bb] = 1
                        # topo_cp[coor_bb] = 7 + dim_int
                        no_2 = no_2 + 1

                    elif not b_exists and d_exists:
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_dd] = 0
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_3 = no_3 + 1
        t3 =time.time()
        # print('2. post processing: ', t3-t2)
        return topo_cp_weight_map, topo_cp_ref_map, betti, batch, cgm_idx

    def save_nii(self, name_list, *map_all):
        '''map: [batch, x, y, z]'''
        for i, map in enumerate(map_all):
            if len(map.shape) == 4:
                map = map[0, ...]
            elif len(map.shape) == 5:
                map = map[0, 0, ...]

            if type(map) == torch.Tensor:
                map = map.cpu().detach().numpy()
            # elif type(map) == np.ndarray:
            #     map = map

            if map.dtype == np.bool:
                map = map.astype(np.float32)

            map = sitk.GetImageFromArray(map)
            sitk.WriteImage(map, name_list[i])
            

    def get_arg_map_range(self, argmap, batch, cgm_idx):
        # argmap = argmap.cpu().detach().numpy()
        x = None
        y = None
        z = None
        # for batch in range(argmap.shape[0]):
        index = np.argwhere(argmap == 1)
        if index.shape[0] !=0:
            z = [index[:, 0].min(), index[:, 0].max()]
            y = [index[:, 1].min(), index[:, 1].max()]
            x = [index[:, 2].min(), index[:, 2].max()]
        return z,y,x,batch, cgm_idx

    def onehot(self, map):
        '''map: torch.tensor in shape [batch, c, x, y, z] '''
        map_argmax = torch.argmax(map, dim=1)
        map_argmax_np = map_argmax.cpu().detach().numpy()
        map_np = map.cpu().detach().numpy()
        labels_all = np.unique(map_argmax_np)
        labels_all.sort()
        lab_array_one_hot = np.zeros_like(map_np)
        for idx, lab in enumerate(labels_all):
                # dist[idx] = (lab_array == lab).astype(float).sum()
                lab_array_one_hot[:, idx, ...] = map_argmax_np == lab
        return lab_array_one_hot

    def forward(self, map, device, cgm_dims=None, topo_size=160, isBinary=False):
        '''for the gt should be betti (0,0,1)'''
        if cgm_dims is not None:
            batch_size, _, xx, yy, zz = map.shape

            cgm_no = len(cgm_dims)
            topo_cp_weight_map = np.zeros((batch_size, cgm_no, xx, yy, zz))
            topo_cp_ref_map = np.zeros((batch_size, cgm_no, xx, yy, zz))
            betti = np.zeros((batch_size, cgm_no, 3))

            argmap = torch.argmax(map, dim=1)
            t1 = time.time()
            '''step_1: calculate the sub-patch for calculating the betti'''
            '''z in the size of [batch, cgm_no, 2]'''
            z = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            y = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            x = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            res_all_1 = []
            pool_1 = mlp.Pool(2*batch_size)
            for batch in range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    # idx = np.s_[batch, cgm_dim, :]
                    mmap = (argmap[batch, ...] == cgm_dim).float().cpu().detach().numpy()
                    if mmap.max() != 1:
                        print('hi')
                    res = pool_1.apply_async(func=self.get_arg_map_range, args=(mmap, batch, cgm_idx))
                    res_all_1.append(res)
                    #
            pool_1.close()
            pool_1.join()
            t2 = time.time()
            print_time = False
            if print_time:
                print('step1.1: ', t2 - t1)
            '''get the result'''
            assert len(res_all_1) == batch_size * cgm_no
            for i, res in enumerate(res_all_1):
                z_t, y_t, x_t, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[batch, cgm_idx, :]
                z[idx], y[idx], x[idx] = z_t, y_t, x_t

            t3 = time.time()
            if print_time:
                print('step1.2: ', t3 - t2)
            '''step_2: calculate betti based on the sub-patch'''
            res_all_2 = []

            pool_2 = mlp.Pool(batch_size*cgm_no)        #
            for batch in  range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    idx = np.s_[batch,
                                cgm_dim,
                                z[batch, cgm_idx, 0]:z[batch, cgm_idx, 1],
                                y[batch, cgm_idx, 0]:y[batch, cgm_idx, 1],
                                x[batch, cgm_idx, 0]:x[batch, cgm_idx, 1]]
                    if isBinary:
                        map_onehot = self.onehot(map)
                        input = map_onehot[idx]
                    else:
                        input = map[idx].cpu().detach().numpy()
                    # input[input>0.5] = 1
                    # input[input<0.5] =0
                    res = pool_2.apply_async(func=self.get_topo_loss, args=(input, batch, cgm_idx))
                    res_all_2.append(res)

            pool_2.close()
            pool_2.join()
            t4 = time.time()
            if print_time:
                print('step2.1: ', t4 - t3)
            '''get the result'''
            assert len(res_all_2) == batch_size * cgm_no
            for i, res in enumerate(res_all_2):
                weight_map, ref_map, bbet, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[batch,
                              cgm_idx,
                              z[batch, cgm_idx, 0]:z[batch, cgm_idx, 1],
                              y[batch, cgm_idx, 0]:y[batch, cgm_idx, 1],
                              x[batch, cgm_idx, 0]:x[batch, cgm_idx, 1]]
                topo_cp_weight_map[idx], topo_cp_ref_map[idx], betti[batch, cgm_idx, :] = weight_map, ref_map, bbet,
            t5 = time.time()
            if print_time:
                print('step2.2: ', t5-t4)
        else:
            assert not isBinary, 'when cgm_dim is not given, the input is 4d tensor, not supprot argmax calculation'
            topo_cp_weight_map = np.zeros(map.shape)
            topo_cp_ref_map = np.zeros(map.shape)
            betti = np.zeros((map.shape[0], 3))
            for batch in range(map.shape[0]):
                for z in range(0, map.shape[1], topo_size):
                    for y in range(0, map.shape[2], topo_size):
                        for x in range(0, map.shape[3], topo_size):
                            if not isBinary:
                                map_patch = map[batch, z:min(z+topo_size, map.shape[1]), y:min(y+topo_size, map.shape[2]), x:min(x+topo_size, map.shape[3])]
                            else:
                                map_argamax = map.unsqueeze(dim=1)
                                argma
                                map_patch = map_[batch, z:min(z+topo_size, map.shape[1]), y:min(y+topo_size, map.shape[2]), x:min(x+topo_size, map.shape[3])]

                            # self.save_labelmap_thres(map_batch, 0.05)
                            topo_cp_weight_map[batch, z:min(z+topo_size, map.shape[1]), y:min(y+topo_size, map.shape[2]), x:min(x+topo_size, map.shape[3])], \
                            topo_cp_ref_map[batch, z:min(z+topo_size, map.shape[1]), y:min(y+topo_size, map.shape[2]), x:min(x+topo_size, map.shape[3])], bb\
                                = self.get_topo_loss(map_patch.cpu().detach().numpy())
                            betti[batch, :] = bb

        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float32).to(device)
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float32).to(device)

        loss_topo = torch.zeros(cgm_no, dtype=torch.float32).to(device)
        num_points_updating = 0
        for cgm_idx, cgm_dim in enumerate(cgm_dims):
            idx = np.s_[:, cgm_idx, ...]
            idx_for_map = np.s_[:, cgm_dim, ...]
            loss_topo[cgm_idx] = 1 / (map.shape[0] * map.shape[2] * map.shape[3]* map.shape[4]) * (((map[idx_for_map] * topo_cp_weight_map[idx]) - topo_cp_ref_map[idx]) ** 2).sum()
            num_points_updating += topo_cp_weight_map[idx].sum()
        betti_return = betti.mean(axis=0)

        return loss_topo, betti_return, num_points_updating/(map.shape[0] * map.shape[2] * map.shape[3]* map.shape[4] * len(cgm_dims))


class topo_loss_2d(nn.Module):
    def __init__(self, package='cripser'):
        super(topo_loss_2d, self).__init__()
        '''package = gudhi (8-connectivity in 2d) or cripser (4-connectivity in 2d) '''
        self.package = package

    def compute_dgm_force_new(self, lh_dgm, gt_dgm, pers_thresh=0, pers_thresh_perfect=0.99, do_return_perfect=False):
        idx_fix_holes = {}
        idx_remove_holes = {}

        for dim in list(lh_dgm.keys()):
            idx_fix_holes.update({dim: []})
            idx_remove_holes.update({dim: []})
            dim_int = int(dim)
            lh_pers = abs(lh_dgm[dim][:, 1] - lh_dgm[dim][:, 0])
            lh_pers_idx_ranked = np.argsort(lh_pers)[::-1]
            lh_n = len(lh_pers)

            if dim in gt_dgm.keys():
                gt_pers = abs(gt_dgm[dim][:, 1] - gt_dgm[dim][:, 0])
                if 0 in gt_pers:        # all background segmentation
                    gt_n = 0    #len(gt_pers) - len(np.where(gt_pers == 0))
                else:
                    gt_n = len(gt_pers)
                    assert np.array_equal(gt_pers, np.ones(gt_n))
            else:
                gt_pers = None
                gt_n = 0

            '''the number of likelihood complex > gt: some of them fixed and some of them removed '''
            if lh_n > gt_n:
                N_holes_2_fix = gt_n
                N_holes_2_remove = lh_n - gt_n

                idx_fix_holes.update({dim: lh_pers_idx_ranked[0:gt_n]})
                idx_remove_holes.update({dim: lh_pers_idx_ranked[gt_n::]})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix
                assert len(idx_remove_holes[dim]) == N_holes_2_remove
            elif lh_n <= gt_n:
                N_holes_2_fix = lh_n
                N_holes_2_remove = 0
                idx_fix_holes.update({dim: lh_pers_idx_ranked})
                assert len(idx_fix_holes[dim]) == N_holes_2_fix

        return idx_fix_holes, idx_remove_holes

    # def recover(self, dim_eff):

    def pre_process(self, info):
        '''info in shape [dim, b, d, b_x, b_y. b_z, d_x, d_y, d_z]'''

        revised_row_array = np.where(info[:, 2] > 1)[0]
        for row in revised_row_array:
            info[row, 2] = 1

        dim_eff_all = np.unique(info[:, 0])
        pd_gt_1 = {}
        bcp_gt_1 = {}
        dcp_gt_1 = {}

        for dim_eff in dim_eff_all:
            idx = info[:, 0] == dim_eff
            pd_gt_1.update({str(int(dim_eff)): info[idx][:, 1:3]})
            bcp_gt_1.update({str(int(dim_eff)): info[idx][:, 3:6]})
            dcp_gt_1.update({str(int(dim_eff)): info[idx][:, 6::]})

        return pd_gt_1, bcp_gt_1, dcp_gt_1

    def check_point_exist(self, r_max, p):
        result = True
        for i, point in enumerate(list(p)):
            if point < 0 or point >= r_max[i]:
                result = False
        return result

    def get_prediction_map(self, pd_lh, map, name):
        root = '/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_5/betti_new/'
        threshold_born = 1 - pd_lh[0]
        threshold_death = 1 - pd_lh[1]

        map_b = torch.zeros(map.shape)
        map_d = torch.zeros(map.shape)
        map_b[map > threshold_born] = 1
        map_d[map > threshold_death] = 1
        self.save_nii([root + x + name + '_p_' + str(threshold_born) + '.nii.gz' for x in ['map_born_', ]], map_b)
        self.save_nii([root + x + name + '_p_' + str(threshold_death) + '.nii.gz' for x in ['map_death_']], map_d)

    def save_B_D_points(self, map, pd_lh, bcp_lh, dcp_lh, no=1000):
        result = np.zeros(map.shape)
        result_inv = np.zeros(map.shape)
        for i in range(no):
            coor_b = tuple(bcp_lh['0'][i].astype(np.int))
            coor_d = tuple(dcp_lh['0'][i].astype(np.int))
            coor_b_inv = tuple(bcp_lh['0'][i].astype(np.int)[::-1])
            coor_d_inv = tuple(dcp_lh['0'][i].astype(np.int)[::-1])
            result[coor_b] = i
            result[coor_d] = -1 * i

            result_inv[coor_b_inv] = i
            result_inv[coor_d_inv] = -1 * i
        self.save_nii(
            ['/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_7/betti/points.nii.gz',
             '/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_7/betti/points_inv.nii.gz'],
            result, result_inv)

    # def reidx_f_gudhi(self, idx, ori_shape):
    #     re_idx = np.zeros(3, dtype=np.uint8)
    #     if len(ori_shape) == 3:
    #         div_0 = ori_shape[1] * ori_shape[2]
    #         re_idx[0] = int(idx // div_0)
    #         mod_0 = idx % div_0
    #
    #         re_idx[1] = int(mod_0 // ori_shape[1])
    #         re_idx[2] = int(mod_0 % ori_shape[1])
    #         assert idx == re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[1] + re_idx[2]
    #     elif len(ori_shape) == 2:
    #         re_idx[0] = int(idx // ori_shape[1])
    #         re_idx[1] = int(idx % ori_shape[1])
    #         assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
    #
    #     # elif len(ori_shape) == 2:
    #     #     re_idx[0] = int(idx // ori_shape[0])
    #     #     re_idx[1] = int(idx % ori_shape[0])
    #     #     assert idx == re_idx[0] * ori_shape[0] + re_idx[1]
    #
    #     return re_idx

    def reidx_f_gudhi(self, idx, ori_shape):
        re_idx = np.zeros(3, dtype=np.uint16)
        reidx_0 = np.array(np.unravel_index(idx, ori_shape, order='C'))
        reidx_1 = np.array(np.unravel_index(idx, ori_shape, order='F'))
        if len(ori_shape) == 3:
            div_0 = ori_shape[1] * ori_shape[2]
            re_idx[0] = int(idx // div_0)
            mod_0 = idx % div_0

            re_idx[1] = int(mod_0 // ori_shape[2])  # updated on 23/09/02 from ori_shape[1] to ori_shape[2]
            re_idx[2] = int(mod_0 % ori_shape[2])
            if idx != re_idx[0] * ori_shape[1] * ori_shape[2] + re_idx[1] * ori_shape[2] + re_idx[
                2]:  # updated on 23/09/02 from re_idx[1] * ori_shape[1] to re_idx[1] * ori_shape[2]
                print('hold on, wrong reidx')
            assert (
                        reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

            # div_0 = ori_shape[1] * ori_shape[0]
            # re_idx[2] = int(idx // div_0)
            # mod_0 = idx % div_0
            #
            # re_idx[1] = int(mod_0 // ori_shape[0])
            # re_idx[0] = int(mod_0 % ori_shape[0])
            # assert idx == re_idx[2] * ori_shape[1] * ori_shape[0] + re_idx[1] * ori_shape[0] + re_idx[0]
            # assert (reidx_1 == re_idx).all(), 'the reindex is not following mode F'
            #
            # if not (re_idx[0] <= ori_shape[0] and re_idx[1] <= ori_shape[1] and re_idx[2] <= ori_shape[2]):
            #     print('hi')

        elif len(ori_shape) == 2:
            re_idx[0] = int(idx // ori_shape[1])
            re_idx[1] = int(idx % ori_shape[1])
            assert idx == re_idx[0] * ori_shape[1] + re_idx[1]
            assert (
                        reidx_0 == re_idx).all(), 'Not C type indexing. The right one should be inverse map shape when establish gd cubicalcomplex and use C type indexing.'

            # re_idx[1] = int(idx // ori_shape[0])
            # re_idx[0] = int(idx % ori_shape[0])
            # assert idx == re_idx[1] * ori_shape[0] + re_idx[0]
        return re_idx  # re_idx

    def get_info_gudhi(self, map):
        cc = gd.CubicalComplex(dimensions=map.shape[::-1], top_dimensional_cells=1 - map.flatten())
        ph = cc.persistence()
        # betti_2 = cc.persistent_betti_numbers(from_value=1, to_value=0)
        x = cc.cofaces_of_persistence_pairs()

        '''3.1 get birth and death point coordinate from gudhi, and generate info array'''
        info_gudhi = np.zeros((len(ph), 9))
        # x will lack one death point where the filtration is inf
        '''3.1.1 manually write the inf death point'''
        reidx_birth_0 = self.reidx_f_gudhi(x[1][0][0], map.shape)
        if len(map.shape) == 2:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1]]
        elif len(map.shape) == 3:
            birth_filtration = 1 - map[reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2]]

        info_gudhi[0, :] = [0, birth_filtration, 1,
                            reidx_birth_0[0], reidx_birth_0[1], reidx_birth_0[2],
                            0, 0, 0]
        idx_row = 1
        for dim in range(len(x[0])):
            for idx in range(x[0][dim].shape[0]):
                idx_brith, idx_death = x[0][dim][idx]
                reidx_birth = self.reidx_f_gudhi(idx_brith, map.shape)
                reidx_death = self.reidx_f_gudhi(idx_death, map.shape)

                if len(map.shape) == 2:
                    if reidx_birth[0]>=map.shape[0] or reidx_birth[1]>=map.shape[1] or reidx_death[0]>=map.shape[0] or reidx_death[1]>=map.shape[1]:
                        print('hold')
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1]]
                elif len(map.shape) == 3:
                    birth_filtration = 1 - map[reidx_birth[0], reidx_birth[1], reidx_birth[2]]
                    death_filtration = 1 - map[reidx_death[0], reidx_death[1], reidx_death[2]]
                else:
                    assert False, 'wrong input dimension!'

                info_gudhi[idx_row, :] = [dim, birth_filtration, death_filtration,
                                          reidx_birth[0], reidx_birth[1], reidx_birth[2],
                                          reidx_death[0], reidx_death[1], reidx_death[2]]
                idx_row += 1

        return info_gudhi

    def get_topo_loss(self, map, gt, batch, cgm_idx):

        t1 = time.time()
        betti = np.zeros((1, 3))
        if self.package == 'cripser':
            info_lh = cripser.computePH(1 - map, maxdim=2)
            info_lh_gt = cripser.computePH(1 - gt, maxdim=2)
        elif self.package == 'gudhi':
            info_lh = self.get_info_gudhi(map)
            info_gt = self.get_info_gudhi(gt)
        else:
            assert False, print('wrong package to calculate topology')
        t2 = time.time()

        pd_lh, bcp_lh, dcp_lh = self.pre_process(info_lh)
        pd_gt, bcp_gt, dcp_gt = self.pre_process(info_gt)
        for i in range(3):
            betti[0, i] = len(pd_lh[str(i)]) if str(i) in list(pd_lh.keys()) else 0

        '''check the topology with the top ranked ph length, only in dim 0'''
        deta = pd_lh['0'][:, 1] - pd_lh['0'][:, 0]
        rank = np.argsort(deta)
        # for i in range(5):
        #     self.get_prediction_map(pd_lh['0'][rank[-1 * i]], map, 'rank_' + str(i))

        '''show top 10 coordinate points that has the minimal born and death (prob ->1)'''
        # self.save_B_D_points(map, pd_lh, bcp_lh, dcp_lh)
        # self.save_nii( ['/vol/biomedic3/ll1320/dhcp/code/pymira-master/pymira/apps/atlas/output/itn_7/betti/ori.nii.gz'], map)
        # no_gt = int(0.05* len(pd_lh['0'][:, 1] )) if int(0.05* len(pd_lh['0'][:, 1] ))>1 else 1
        # no_gt = 1
        # a = np.zeros([no_gt, 2])
        # a[:, 1] = 1
        # pd_gt = {'0': a}  # {'0': a, '2':a}        # np.array([[0, 1]])

        idx_holes_to_fix, idx_holes_to_remove = self.compute_dgm_force_new(pd_lh, pd_gt, pers_thresh=0)
        '''
        topo_cp value map:
        0： background
        1, 2, 3：point to fix (born: force it to 0)
        4, 5, 6: point to fix (death: force it to 1)
        7, 8, 9: point to remove (born: force it to the prob of death)
        10,11,12: point to remove (death: force it to the prob of born)
        '''
        topo_size = list(map.shape)
        topo_cp_weight_map = np.zeros(map.shape)
        topo_cp_ref_map = np.zeros(map.shape)

        for dim in ['0', '1']:
            dim_int = int(dim)
            if dim in list(idx_holes_to_fix.keys()):
                for hole_indx in idx_holes_to_fix[dim]:
                    if self.check_point_exist(topo_size, bcp_lh[dim][hole_indx][0:2]):
                        # if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_bb = (coor_b[0], coor_b[1])
                        topo_cp_weight_map[coor_bb] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_bb] = 0

                        # self.get_prediction_map(pd_lh[dim][hole_indx], map)

                        # topo_cp[coor_bb] = 1 + dim_int

                    # if(y+int(dcp_lh[hole_indx][0]) < et_dmap.shape[2] and x+int(dcp_lh[hole_indx][1]) < et_dmap.shape[3]):
                    if self.check_point_exist(topo_size, dcp_lh[dim][hole_indx][0:2]):
                        # if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                        coor_dd = (coor_d[0], coor_d[1])
                        # topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1  # push death to 1 i.e. max death prob or likelihood
                        # topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 1
                        topo_cp_weight_map[coor_dd] = 1  # push birth to 0 i.e. min birth prob or likelihood
                        topo_cp_ref_map[coor_dd] = 1

                        # topo_cp[coor_dd] = 4 + dim_int
            no_1 = 0
            no_2 = 0
            np_3 = 0
            if dim in list(idx_holes_to_remove.keys()):
                for hole_indx in idx_holes_to_remove[dim]:
                    coor_b = [int(bcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_bb = (coor_b[0], coor_b[1])

                    coor_d = [int(dcp_lh[dim][hole_indx][ii]) for ii in range(3)]
                    coor_dd = (coor_d[0], coor_d[1])

                    b_exists = self.check_point_exist(topo_size, bcp_lh[dim][hole_indx][0:2])
                    d_exists = self.check_point_exist(topo_size, dcp_lh[dim][hole_indx][0:2])
                    if b_exists and d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_bb] = map[coor_dd]
                        topo_cp_ref_map[coor_dd] = map[coor_bb]
                        # topo_cp[coor_bb] = 7 + dim_int
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_1 = no_1 + 1
                    elif b_exists and not d_exists:
                        topo_cp_weight_map[coor_bb] = 1
                        topo_cp_ref_map[coor_bb] = 1
                        # topo_cp[coor_bb] = 7 + dim_int
                        no_2 = no_2 + 1

                    elif not b_exists and d_exists:
                        topo_cp_weight_map[coor_dd] = 1
                        topo_cp_ref_map[coor_dd] = 0
                        # topo_cp[coor_dd] = 10 + dim_int
                        no_3 = no_3 + 1
        t3 = time.time()
        # print('1. post processing: ', t3-t2)
        return topo_cp_weight_map, topo_cp_ref_map, betti, batch, cgm_idx

    def save_nii(self, name_list, *map_all):
        '''map: [batch, x, y, z]'''
        for i, map in enumerate(map_all):
            if len(map.shape) == 4:
                map = map[0, ...]
            elif len(map.shape) == 5:
                map = map[0, 0, ...]

            if type(map) == torch.Tensor:
                map = map.cpu().detach().numpy()
            # elif type(map) == np.ndarray:
            #     map = map

            if map.dtype == np.bool:
                map = map.astype(np.float32)

            map = sitk.GetImageFromArray(map)
            sitk.WriteImage(map, name_list[i])

    def get_arg_map_range(self, argmap, batch, cgm_idx):
        # argmap = argmap.cpu().detach().numpy()
        x = None
        y = None
        # for batch in range(argmap.shape[0]):
        index = np.argwhere(argmap == 1)
        if index.shape[0] != 0:
            y = [index[:, 0].min(), index[:, 0].max()]
            x = [index[:, 1].min(), index[:, 1].max()]
        return y, x, batch, cgm_idx

    def forward(self, map, gt, device, cgm_dims=None, topo_size=256):
        '''for the gt should be betti (0,0,1)'''
        if cgm_dims is not None:
            batch_size, _, xx, yy = map.shape

            cgm_no = len(cgm_dims)
            topo_cp_weight_map = np.zeros((batch_size, cgm_no, xx, yy))
            topo_cp_ref_map = np.zeros((batch_size, cgm_no, xx, yy))
            betti = np.zeros((batch_size, cgm_no, 3))

            argmap = torch.argmax(map, dim=1)
            t1 = time.time()
            '''step_1: calculate the sub-patch for calculating the betti'''
            '''z in the size of [batch, cgm_no, 2]'''
            # z = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            y = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            x = np.zeros((batch_size, cgm_no, 2), dtype=np.int64)
            res_all_1 = []
            pool_1 = mlp.Pool(2 * batch_size)
            for batch in range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    # idx = np.s_[batch, cgm_dim, :]
                    mmap = (argmap[batch, ...] == cgm_dim).float().cpu().detach().numpy()
                    res = pool_1.apply_async(func=self.get_arg_map_range, args=(mmap, batch, cgm_idx))
                    res_all_1.append(res)
                    #
            pool_1.close()
            pool_1.join()
            t2 = time.time()
            print_time = False
            if print_time:
                print('step1.1: ', t2 - t1)
            '''get the result'''
            assert len(res_all_1) == batch_size * cgm_no
            for i, res in enumerate(res_all_1):
                y_t, x_t, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[batch, cgm_idx, :]
                y[idx], x[idx] = y_t, x_t

            t3 = time.time()
            if print_time:
                print('step1.2: ', t3 - t2)
            '''step_2: calculate betti based on the sub-patch'''
            res_all_2 = []

            pool_2 = mlp.Pool(batch_size * cgm_no)  #
            for batch in range(batch_size):
                for cgm_idx, cgm_dim in enumerate(cgm_dims):
                    idx = np.s_[batch,
                          cgm_dim,
                          y[batch, cgm_idx, 0]:y[batch, cgm_idx, 1],
                          x[batch, cgm_idx, 0]:x[batch, cgm_idx, 1]]
                    input = map[idx].cpu().detach().numpy()
                    # input[input>0.5] = 1
                    # input[input<0.5] =0
                    res = pool_2.apply_async(func=self.get_topo_loss, args=(input, batch, cgm_idx))
                    res_all_2.append(res)

            pool_2.close()
            pool_2.join()
            t4 = time.time()
            if print_time:
                print('step2.1: ', t4 - t3)
            '''get the result'''
            assert len(res_all_2) == batch_size * cgm_no
            for i, res in enumerate(res_all_2):
                weight_map, ref_map, bbet, batch, cgm_idx = res.get()
                assert batch * cgm_no + cgm_idx == i
                idx = np.s_[batch,
                      cgm_idx,
                      y[batch, cgm_idx, 0]:y[batch, cgm_idx, 1],
                      x[batch, cgm_idx, 0]:x[batch, cgm_idx, 1]]
                topo_cp_weight_map[idx], topo_cp_ref_map[idx], betti[batch, cgm_idx, :] = weight_map, ref_map, bbet,
            t5 = time.time()
            if print_time:
                print('step2.2: ', t5 - t4)
        else:
            topo_cp_weight_map = np.zeros(map.shape)
            topo_cp_ref_map = np.zeros(map.shape)
            betti = np.zeros((map.shape[0], 3))
            for batch in range(map.shape[0]):
                for y in range(0, map.shape[2], topo_size):
                    for x in range(0, map.shape[3], topo_size):
                        map_patch = map[batch, 0,
                                    y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])]
                        gt_patch = gt[batch, 0,
                                    y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])]
                        # self.save_labelmap_thres(map_batch, 0.05)
                        topo_cp_weight_map[batch, 0,
                        y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])], \
                        topo_cp_ref_map[batch, 0,
                        y:min(y + topo_size, map.shape[2]), x:min(x + topo_size, map.shape[3])], bb, _, __ \
                            = self.get_topo_loss(map_patch.cpu().detach().numpy(), gt_patch.cpu().detach().numpy(), batch, 0)
                        betti[batch, :] = bb

        topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float32).to(device)
        topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float32).to(device)
        if cgm_dims is None:
            cgm_dims = [0]
            cgm_no = 1
        loss_topo = torch.zeros(cgm_no, dtype=torch.float32).to(device)
        num_points_updating = 0

        for cgm_idx, cgm_dim in enumerate(cgm_dims):
            idx = np.s_[:, cgm_idx, ...]
            idx_for_map = np.s_[:, cgm_dim, ...]
            loss_topo[cgm_idx] = 1 / (map.shape[0] * map.shape[2] * map.shape[3]) * (
                        ((map[idx_for_map] * topo_cp_weight_map[idx]) - topo_cp_ref_map[idx]) ** 2).sum()
            num_points_updating += topo_cp_weight_map[idx].sum()
        betti_return = betti.mean(axis=0)

        return loss_topo, betti_return, num_points_updating / (
                    map.shape[0] * map.shape[2] * map.shape[3] * len(cgm_dims)), topo_cp_weight_map, topo_cp_ref_map


class hausdorff_loss(nn.Module):
    def __init__(self, device):
        super(hausdorff_loss, self).__init__()
        self.device = device

    def get_dist_map(self, mmap, label_inner, label_outer):
        mmap_np = mmap.cpu().numpy()
        level_zero_outer = np.zeros(list(mmap.shape))
        boundary_tensor = torch.zeros(list(mmap.shape), dtype=torch.float32).to(self.device)
        tdm_tensor = torch.zeros(list(mmap.shape), dtype=torch.float32).to(self.device)

        for q in label_outer:  # including label 0
            level_zero_outer[mmap_np == q] = 1

        for k in label_inner:
            level_zero_outer[mmap_np == k] = 0

        '''directly calculate the signed distance function'''
        for batch in range(mmap.shape[0]):
            tdm_outer = ndimage.distance_transform_edt(level_zero_outer[batch, ...])
            boundary = ndimage.distance_transform_edt(1 - level_zero_outer[batch, ...])
            boundary[boundary != 1] = 0
            outer = tdm_outer + boundary  # boundary = 1, outer= tdm value
            level_zero_inner = np.zeros(list(mmap.shape)[1::])
            level_zero_inner[outer == 0] = 1
            tsd_inner = -1 * ndimage.distance_transform_edt(level_zero_inner)
            tsd = tsd_inner + tdm_outer
            boundary_tensor[batch, ...] = torch.from_numpy(boundary).float()
            tdm_tensor[batch, ...] = torch.from_numpy(tsd).float()

        return boundary_tensor, tdm_tensor

    def forward(self, map, d_mean=2):
        '''map is a 4d tensor [batch, x, y ,z]'''

        inner_boundary, phi_inner = self.get_dist_map(map, label_inner=[3,5,6,7,8,9], label_outer=[2, 1, 4, 0])
        outer_boundary, phi_outer = self.get_dist_map(map, label_inner=[3,5,6,7,8,9, 2], label_outer=[1, 4, 0])

        hausdorff_inner = torch.abs(inner_boundary * phi_outer)
        hausdorff_outer = torch.abs(outer_boundary * phi_inner)

        loss = (hausdorff_inner.sum()/inner_boundary.sum() - d_mean) ** 2 + (hausdorff_outer.sum() / outer_boundary.sum() - d_mean) ** 2
        hausdorff_inner_mean = hausdorff_inner.sum()/inner_boundary.sum()
        hausdorff_inner_max = hausdorff_inner.max()
        hausdorff_outer_mean = hausdorff_outer.sum()/outer_boundary.sum()
        hausdorff_outer_max = hausdorff_outer.max()

        return hausdorff_inner_mean, hausdorff_inner_max, hausdorff_outer_mean, hausdorff_outer_max, loss


class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.
        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))
        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)



def l2reg_loss(u):
    """L2 regularisation loss"""
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def bending_energy_loss(u):
    """Bending energy regularisation loss"""
    derives = []
    ndim = u.size()[1]
    # 1st order
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    # 2nd order
    derives2 = []
    for i in range(ndim):
        derives2 += [finite_diff(derives[i], dim=i)]  # du2xx, du2yy, (du2zz)
    derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=1)]  # du2dxy
    if ndim == 3:
        derives2 += [math.sqrt(2) * finite_diff(derives[0], dim=2)]  # du2dxz
        derives2 += [math.sqrt(2) * finite_diff(derives[1], dim=2)]  # du2dyz

    assert len(derives2) == 2 * ndim
    loss = torch.cat(derives2, dim=1).pow(2).sum(dim=1).mean()
    return loss


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Input shape (N, ndim, *sizes), mode='foward', 'backward' or 'central'"""
    assert type(x) is torch.Tensor
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    if mode == "central":
        # TODO: implement central difference by 1d conv or dialated slicing
        raise NotImplementedError("Finite difference central difference mode")
    else:  # "forward" or "backward"
        # configure padding of this dimension
        paddings = [[0, 0] for _ in range(ndim)]
        if mode == "forward":
            # forward difference: pad after
            paddings[dim][1] = 1
        elif mode == "backward":
            # backward difference: pad before
            paddings[dim][0] = 1
        else:
            raise ValueError(f'Mode {mode} not recognised')

        # reverse and join sublists into a flat list (Pytorch uses last -> first dim order)
        paddings.reverse()
        paddings = [p for ppair in paddings for p in ppair]

        # pad data
        if boundary == "Neumann":
            # Neumann boundary condition
            x_pad = F.pad(x, paddings, mode='replicate')
        elif boundary == "Dirichlet":
            # Dirichlet boundary condition
            x_pad = F.pad(x, paddings, mode='constant')
        else:
            raise ValueError("Boundary condition not recognised.")

        # slice and subtract
        x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
                 - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

        return x_diff

def decide_simple_point_2D(gt, x, y):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x - 1:x + 2, y - 1:y + 2]

    ## check local topology

    number_fore, _ = cv2.connectedComponents(patch, 4)
    number_back, _ = cv2.connectedComponents(1 - patch, 8)

    label = (number_fore - 1) * (number_back - 1)

    # try:
    #     patch[1][1] = 0
    #     number_fore, label_fore = cv2.connectedComponents(patch, 4)
    #     label_fore_4 = np.unique([label_fore[0,1], label_fore[1,0], label_fore[2,1], label_fore[1,2]])

    #     patch_reverse = 1 - patch
    #     patch_reverse[1][1] = 0
    #     number_back, label_back = cv2.connectedComponents(patch_reverse, 8)
    #     label_back_8 = np.unique([label_back[0,0], label_back[0,1], label_back[0,2], label_back[1,0], label_back[1,2], label_back[2,0], label_back[2,1], label_back[2,2]])
    #     label = len(np.nonzero(label_fore_4)[0]) * len(np.nonzero(label_back_8)[0])
    # except:
    #         label = 0
    #         pass

    ## flip the simple point
    if (label == 1):
        gt[x, y] = 1 - gt[x, y]

    return gt


def decide_simple_point_3D(gt, x, y, z):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]

    ## check local topology
    if patch.shape[0] != 0 and patch.shape[1] != 0 and patch.shape[2] != 0:
        try:
            _, number_fore = cc3d.connected_components(patch, 6, return_N=True)
            _, number_back = cc3d.connected_components(1 - patch, 26, return_N=True)
        except:
            number_fore = 0
            number_back = 0
            pass
        label = number_fore * number_back

        ## flip the simple point
        if (label == 1):
            gt[x, y, z] = 1 - gt[x, y, z]

    return gt


def update_simple_point(distance, gt):
    non_zero = np.nonzero(distance)
    # indice = np.argsort(-distance, axis=None)
    indice = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

    if len(gt.shape) == 2:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_2D(gt, x, y)
    else:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]
            z = indice[2][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_3D(gt, x, y, z)
    return gt


def warping_loss(y_pred, y_gt):
    """
    Calculate the warping loss of the predicted image and ground truth image
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative

    # if ()
    loss = 0
    # soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
    # train_loss_func = DC_and_CE_loss(soft_dice_args, {})
    # sdl = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
    # ce_loss = RobustCrossEntropyLoss()

    if (len(y_pred.shape) == 4):
        B, C, H, W = y_pred.shape
        # pre = y_pred
        # pre = softmax_helper(y_pred)
        # import pdb;
        # pdb.set_trace()
        pre = torch.zeros_like(y_pred)
        pre[y_pred >= 0.5] =1
        pre[y_pred < 0.5] = 0
        pre = pre.squeeze(dim=1)
        # pre = torch.argmax(y_pred, dim=1)


        y_gt = torch.unsqueeze(y_gt[:, 0, :, :], dim=1)
        gt = torch.squeeze(y_gt, dim=1)

        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy()
        gt_copy = gt.copy()

        critical_points = np.zeros((B, H, W))
        for i in range(B):
            false_positive = ((pre_copy[i, :, :] - gt_copy[i, :, :]) == 1).astype(int)
            false_negative = ((gt_copy[i, :, :] - pre_copy[i, :, :]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance_gt = ndimage.distance_transform_edt(
                gt_copy[i, :, :]) * false_negative  # shrink gt while keep connected
            false_positive_distance_gt = ndimage.distance_transform_edt(
                1 - gt_copy[i, :, :]) * false_positive  # grow gt while keep unconnected
            gt_warp = update_simple_point(false_negative_distance_gt, gt_copy[i, :, :])
            gt_warp = update_simple_point(false_positive_distance_gt, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(
                pre_copy[i, :, :]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(
                1 - pre_copy[i, :, :]) * false_negative  # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i, :, :])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i, :, :] = np.logical_or(np.not_equal(pre[i, :, :], gt_warp),
                                                     np.not_equal(gt[i, :, :], pre_warp)).astype(int)
    else:
        B, C, H, W, Z = y_pred.shape
        pre = y_pred
        # pre = softmax_helper(y_pred)
        pre = torch.argmax(pre, dim=1)
        y_gt = torch.unsqueeze(y_gt[:, 0, :, :, :], dim=1)
        gt = torch.squeeze(y_gt, dim=1)

        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy()
        gt_copy = gt.copy()

        critical_points = np.zeros((B, H, W, Z))
        for i in range(B):
            false_positive = ((pre_copy[i, :, :, :] - gt_copy[i, :, :, :]) == 1).astype(int)
            false_negative = ((gt_copy[i, :, :, :] - pre_copy[i, :, :, :]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance = ndimage.distance_transform_edt(gt_copy[i, :, :, :]) * false_negative
            false_positive_distance = ndimage.distance_transform_edt(1 - gt_copy[i, :, :, :]) * false_positive
            gt_warp = update_simple_point(false_negative_distance, gt_copy[i, :, :, :])
            gt_warp = update_simple_point(false_positive_distance, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(
                pre_copy[i, :, :, :]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(
                1 - pre_copy[i, :, :, :]) * false_negative  # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i, :, :, :])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i, :, :] = np.logical_or(np.not_equal(pre[i, :, :, :], gt_warp),
                                                     np.not_equal(gt[i, :, :, :], pre_warp)).astype(int)

    pred_prime = y_pred * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda()
    gt_prime = (y_gt * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda()).detach()
    loss = F.binary_cross_entropy(pred_prime, gt_prime) * len(np.nonzero(critical_points)[0])
    # loss = ce_loss(y_pred * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda(),
    #                y_gt * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda()) * len(
    #     np.nonzero(critical_points)[0])
    # print()
    return loss, len(np.nonzero(critical_points)[0])