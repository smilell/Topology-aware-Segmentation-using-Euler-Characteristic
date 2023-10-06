import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import pickle
import random
import cv2
import time

from utils.topo import get_topo_gudhi
from utils.utils import *
from process_batch import smoothing_2d

def filter(x, thre=0.5):
    x[x > thre] = 1
    x[x<thre]=0
    return x

class ImageSegmentation_cremi(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, args, img_data, seg_data, softmax_data=None, weight_grad=None, csv_file_betti=None,
                 patch_size_list=None, stride_size=None, crop=True, centerCrop=False, cropSize=None,
                 flip=False, multi=1, subdata=False, weight=None,
                 patch_info_path=None, mode=None):

        self.img_data = img_data
        self.seg_data = seg_data

        # only for postNet
        self.softmax_data = softmax_data
        self.weight_grad = weight_grad

        self.crop = crop
        self.centerCrop = centerCrop
        self.cropSize = cropSize
        self.flip = flip
        self.subdata = subdata
        self.samples = []
        self.results =[]
        self.info = {}      # save the GA and its corresponding sample index
        self.args = args
        self.dist = {}
        self.weight = weight

        self.betti_file_dir = csv_file_betti
        if self.betti_file_dir is not None and not os.path.exists(self.betti_file_dir):
            os.mkdir(self.betti_file_dir)       # input a betti path but doesn't exist the path
        self.patch_size_list = patch_size_list
        self.stride_size = stride_size
        self.post = args.phase in ['train_post', 'test_post',]

        # only to generate weight map for spatial loss
        self.patch_info = patch_info_path
        self.patch_mode = mode
        # self.input_argmax = args.input_argmax


        for n in range(multi):
            data_list = os.listdir(self.img_data)
            data_list = [x for x in data_list if x.endswith('png')]
            if self.subdata:
                data_list = [x for x in data_list if x.startswith('sample_A')]

            data_list.sort()

            for idx, img_name in enumerate(data_list):
                if True:#   and idx <=10:
                    img_path = os.path.join(self.img_data, img_name)        # sample_A_001.jpg
                    img_base_name = img_name.split('.')[0]
                    seg_name = img_base_name + '_boundary.png'              # specific for cremi dataset
                    seg_path = os.path.join(self.seg_data, seg_name)        # 'gif'

                    assert os.path.exists(seg_path), 'No segmentation data in ' + seg_path

                    if self.post:
                        softmax_name_choices = [img_base_name + x for x in
                                                ['_softmax.jpg', '.jpg', '.png', '_softmax.png']]
                        weight_name_choices = [img_base_name + x for x in
                                               ['_weight.jpg', '.jpg', '.png', '_weight.png']]

                        for d in softmax_name_choices:
                            if os.path.exists(os.path.join(self.softmax_data, d)):
                                softmax_path = os.path.join(self.softmax_data, d)
                                break
                        for d in weight_name_choices:
                            if os.path.exists(os.path.join(self.weight_grad, d)):
                                weight_grad_path = os.path.join(self.weight_grad, d)
                                break

                        # softmax_path = os.path.join(self.softmax_data, softmax_name)
                        # weight_grad_path = os.path.join(self.weight_grad, weight_name)
                        assert os.path.exists(softmax_path), 'No softmax data from stage 1 saved in' + softmax_path
                        assert os.path.exists(weight_grad_path), 'No euler visualization results from stage 1 saved in' + weight_grad_path
                    else:
                        softmax_path = None
                        weight_grad_path = None

                    if self.patch_info is not None:
                        patch_info_path = os.path.join(self.patch_info, img_base_name + '.pkl')
                    else:
                        patch_info_path = None

                    sample = {'image': img_path,
                              'labelmap': seg_path,
                              'softmax': softmax_path,
                              'weight_grad': weight_grad_path,
                              'patch_info': patch_info_path,
                              }

                    if self.betti_file_dir is not None:
                        betti_path = os.path.join(self.betti_file_dir, img_base_name + '_betti.pkl')
                        sample.update({'betti': betti_path})
                    else:
                        sample.update({'betti': None})



                    self.samples.append(sample)


                else:
                    break

            # result = self.get_data_dict_by_dir(img_path, seg_path, mask_path)
            # self.results.append(result)


    def __len__(self):
        return len(self.samples)

    def read_onehot_labelmap(self, file_path):
        labelmap_onehot = sitk.ReadImage(file_path, sitk.sitkVectorFloat32)
        labelmap_sitk = labelmap_onehot
        labelmap_onehot = torch.from_numpy(sitk.GetArrayFromImage(labelmap_onehot)).permute(3, 0, 1,2)
        return labelmap_onehot, labelmap_sitk


    def calculate_gt_betti(self, gt, patch_size_list=[8, 16, 32, 64, 128], stride_size=8):
        betti, _ = get_topo_gudhi(gt)          #gt.shape should be 2d???
        betti_map_list = []
        for i, patch_size in enumerate(patch_size_list):
            patch_no = int((gt.shape[-1] - patch_size) / stride_size + 1)
            betti_map = np.zeros((2, patch_no, patch_no))       # betti 0 and betti 1

            for x_idx in range(patch_no):
                for y_idx in range(patch_no):
                    xmin = x_idx * stride_size
                    ymin = y_idx * stride_size
                    xmax = xmin + patch_size
                    ymax = ymin + patch_size
                    gt_patch = gt[xmin: xmax, ymin: ymax]
                    betti_pred, _ = get_topo_gudhi(gt_patch)
                    betti_map[0, x_idx, y_idx] = betti_pred[0]
                    betti_map[1, x_idx, y_idx] = betti_pred[1]

            betti_map_list.append(betti_map)

        betti_dict = {'betti_number': betti,
                      'betti_map_list': betti_map_list,
                      }

        return betti_dict

    def zscore(self, image):
        mean = image.mean()
        std = image.std()
        out = (image - mean) / std
        return out, mean, std

    def zeroone(self, image):
        mmax = image.max()
        mmin = image.min()
        out = (image - mmin)/ (mmax - mmin+ 0.001)
        return out

    def get_seg(self, seg_path, w=None):
        labelmap = cv2.imread(seg_path)
        labelmap = (np.array(labelmap[..., 0]) / 255)
        if weight is not None:
            pass
        else:
            w = [0.1, 0.9]
        weight = torch.zeros(labelmap.shape)
        for i in range(labelmap.shape[0]):
            for j in range(labelmap.shape[1]):
                weight[i, j] = w[int(labelmap[i, j])]
        return labelmap, weight

    def get_img(self, img_path):
        image = cv2.imread(img_path)[..., 0:1]
        image_ori = np.array(image).astype(np.float32)
        image_norm, image_mean, image_std = self.zscore(image_ori)
        return image_ori, image_norm, image_mean, image_std

    def get_img_zero_one(self, img_path):
        # read an image, and map it to [0-1]. e.g. weight_grad and softmax map
        img = cv2.imread(img_path)[..., 0:1]
        img = np.array(img).astype(np.float32)
        img_norm = self.zeroone(img)
        return img_norm

    def get_weight_map(self, good_patch_idx, bad_patch_idx, det, weight_patch, mode):

        '''
        mode =0: bad patch weight = 2, good patch weight = 0.5
        mode =1: bad patch + same number of good patches -> weight = 1, others = 0.1
        '''

        patch_idxs_1 = []
        patch_idxs_2 = []
        if mode == 0:
            patch_idxs_1 = good_patch_idx
            patch_idxs_2 = bad_patch_idx
            weight_1 = 0.5
            weight_2 = 2
        elif mode == 1:
            random.shuffle(good_patch_idx)
            patch_idxs_2 = bad_patch_idx
            weight_1 = 0.1
            weight_2 = 1
            good_patch_no = len(good_patch_idx)
            bad_patch_no = len(bad_patch_idx)
            if good_patch_no >= bad_patch_no:
                good_patch_selected = [good_patch_idx[i] for i in range(bad_patch_no)]
                good_patch_remained = [good_patch_idx[i] for i in range(bad_patch_no, good_patch_no)]
            else:
                good_patch_selected = good_patch_idx
                good_patch_remained = []

            patch_idxs_2 = patch_idxs_2 + good_patch_selected
            patch_idxs_1 = good_patch_remained
        elif mode == 2:
            pass
        else:
            raise 'wrong weight mode'

        for idx, patch_idx in enumerate(patch_idxs_1):
            start_x = patch_idx[0]
            start_y = patch_idx[1]
            weight_patch[start_x:start_x + det, start_y:start_y + det] = weight_1
        for idx, patch_idx in enumerate(patch_idxs_2):
            start_x = patch_idx[0]
            start_y = patch_idx[1]
            weight_patch[start_x:start_x + det, start_y:start_y + det] = weight_2

        return weight_patch

    def get_data_dict_by_dir(self, img_path, seg_path, betti_path=None, softmax_path=None, weight_grad_path=None, patch_info_path=None, patch_mode=None):
        ''' save all the image, labelmap data into one pkl'''
        img_fname = os.path.basename(img_path)
        img_pkl_path = img_path[:-4] + '.pkl'
        seg_pkl_path = seg_path[:-4] + '.pkl'

        if os.path.exists(img_pkl_path):
            with open(img_pkl_path, 'rb') as f:
                image_dict = pickle.load(f)
            image_norm = image_dict['image']
            image_mean = image_dict['image_mean']
            image_std = image_dict['image_std']
        else:
            image_ori, image_norm, image_mean, image_std = self.get_img(img_path)
            with open(img_pkl_path, 'wb') as f:
                pickle.dump({
                    'image': image_norm,
                    'image_mean': image_mean,
                    'image_std': image_std,
                }, f)

        if os.path.exists(seg_pkl_path):
            with open(seg_pkl_path, 'rb') as f:
                seg_dict = pickle.load(f)
            labelmap = seg_dict['labelmap']
            weight = 2 * seg_dict['weight']
        else:
            labelmap, weight = self.get_seg(seg_path, self.weight)
            with open(seg_pkl_path, 'wb') as f:
                pickle.dump({
                    'labelmap': labelmap,
                    'weight': weight,
                }, f)
        # ----------------------------- load data for train_post/ test_post. If run default train stage 1, then it will save pkl file -----------------------------
        if self.post:
            weight_grad_pkl_path = weight_grad_path[:-4] + '.pkl'

            if os.path.exists(weight_grad_pkl_path):
                with open(weight_grad_pkl_path, 'rb') as f:
                    weight_grad = pickle.load(f)
                # print('shape', weight_grad.shape)
                # due to the old version, it could be saved in a dict with the key 'weight', 2d, max=2
                # or saved as a np array, 3d, max=1
                if isinstance(weight_grad, dict):
                    weight_grad = weight_grad['weight']
                    weight_grad = weight_grad/2
                if not self.args.mode3d and len(weight_grad.shape) == 3:
                    weight_grad = weight_grad[:, :, 0]

            else:
                weight_grad = self.get_img_zero_one(weight_grad_path)
                with open(weight_grad_pkl_path, 'wb') as f:
                    pickle.dump(weight_grad, f)


            softmax_coarse_pkl_path = softmax_path[:-4] + '.pkl'
            if os.path.exists(softmax_coarse_pkl_path):
                with open(softmax_coarse_pkl_path, 'rb') as f:
                    softmax_coarse = pickle.load(f)
                if not self.args.mode3d and len(softmax_coarse.shape) == 3:
                    softmax_coarse = softmax_coarse[:, :, 0]
            else:
                softmax_coarse = self.get_img_zero_one(softmax_path)
                with open(softmax_coarse_pkl_path, 'wb') as f:
                    pickle.dump(softmax_coarse, f)

            # if self.input_argmax:
            #     softmax_coarse_copy = np.zeros_like(softmax_coarse)
            #     softmax_coarse_copy[softmax_coarse>=0.5] =1
            #     softmax_coarse = softmax_coarse_copy


        weight_patch = np.zeros_like(labelmap)
        if patch_info_path is not None:
            assert os.path.exists(patch_info_path), 'do not have patch info to re-weight loss'

            with open(patch_info_path, 'rb') as f:
                patch_info_dict = pickle.load(f)
            bad_patch_idx = patch_info_dict['patch_idx_list_bad']
            good_patch_idx = patch_info_dict['patch_idx_list_good']
            weight_binary = patch_info_dict['weight_rank_binary']
            det = patch_info_dict['sub_patch_size']
            weight_patch = self.get_weight_map(good_patch_idx, bad_patch_idx, det, weight_patch, patch_mode)

        if self.crop:
            det = self.args.cropSize#int(image_01.shape[0] / 2)
            if self.centerCrop:
                start_x = int((image_norm.shape[0] - det )/2)
                start_y = int((image_norm.shape[1] - det )/2)
            else:
                start_x = random.randint(0, 512-det)        # int((image_01.shape[0] - det )/2)
                start_y = random.randint(0, 512-det)        # int((image_01.shape[1] - det )/2)   #

            image_norm = image_norm[start_x:start_x + det, start_y: start_y + det, :]
            labelmap = labelmap[start_x:start_x + det, start_y: start_y + det]
            weight_patch = weight_patch[start_x:start_x + det, start_y: start_y + det]
            if self.post:
                weight_grad = weight_grad[start_x:start_x + det, start_y: start_y + det]
                softmax_coarse = softmax_coarse[start_x:start_x + det, start_y: start_y + det]

        if self.flip:
            if random.randint(0, 1):
                r_axis = random.randint(0, 1)
                image_norm = np.flip(image_norm, axis=r_axis).copy()
                labelmap = np.flip(labelmap, axis=r_axis).copy()
                weight_patch = np.flip(weight_patch, axis=r_axis).copy()
                if self.post:
                    weight_grad = np.flip(weight_grad, axis=r_axis).copy()
                    softmax_coarse = np.flip(softmax_coarse, axis=r_axis).copy()


        image_norm = torch.from_numpy(image_norm.transpose(2, 0, 1)).type(torch.float32)
        image_ori =  image_norm * image_std + image_mean
        labelmap_argmax = torch.from_numpy(labelmap).type(torch.float32)  # .unsqueeze(0)
        labelmap_onehot = labelmap_argmax.unsqueeze(dim=0)
        weight_patch_onehot = torch.from_numpy(weight_patch).type(torch.float32).unsqueeze(dim=0)



        result = {'img_fname': img_fname,
                  'image': image_norm,
                  'image_ori': image_ori,
                  'image_info': [image_mean, image_std],
                  'labelmap_argmax': labelmap_argmax,
                  'labelmap_onehot': labelmap_onehot,
                  'weight_patch_onehot': weight_patch_onehot,
                  }
        if self.post:
            weight_grad = torch.from_numpy(weight_grad).type(torch.float32).unsqueeze(dim=0)
            softmax_coarse = torch.from_numpy(softmax_coarse).type(torch.float32).unsqueeze(dim=0)

            result.update({
                'weight_grad': weight_grad,
                'softmax_coarse': softmax_coarse,
            })

        '''process betti data'''
        t3 = time.time()
        if betti_path is not None:      #  either read or write
            if os.path.exists(betti_path):
                with open(betti_path, 'rb') as f:
                    betti_dict = pickle.load(f)
                # print('betti_number in dataset.py, load', betti_dict['betti_number'].dtype)
            else:
                betti_dict = self.calculate_gt_betti(labelmap, self.patch_size_list, self.stride_size)
                with open(betti_path, 'wb') as f:
                    pickle.dump(betti_dict, f)
                print('done', betti_path)
            result.update({'betti_number': betti_dict['betti_number'],
                           'betti_map_list': betti_dict['betti_map_list'],
                           })
        # print(time.time()-t3)
        return result

    def get_data_dict_by_idx(self, item):
        sample = self.samples[item]
        img_path = sample['image']
        seg_path = sample['labelmap']
        softmax_path = sample['softmax']
        weight_grad_path = sample['weight_grad']
        # mask_path = sample['mask']

        result = self.get_data_dict_by_dir(img_path, seg_path, softmax_path=softmax_path, weight_grad_path=weight_grad_path)  #, mask_path)
        return result


    def __getitem__(self, item):
        # print('item:', item)

        # return sample
        """self.sample_index_according_to_age 是所有数据的index按照age的排序"""
        # print('item', item)
        sample = self.samples[item]
        data_1 = self.get_data_dict_by_dir(sample['image'], sample['labelmap'], sample['betti'],
                                           softmax_path=sample['softmax'], weight_grad_path=sample['weight_grad'], patch_info_path=sample['patch_info'], patch_mode=self.patch_mode)

        result= {'image': data_1['image'],
                'image_ori': data_1['image_ori'],
                'labelmap_argmax': data_1['labelmap_argmax'],
                'labelmap_onehot': data_1['labelmap_onehot'],
                'fname': data_1['img_fname'],
                # 'mask': data_1['mask'],
                'image_info': data_1['image_info'],
                # 'weight': data_1['weight'],
                'betti_number': data_1['betti_number'],
                'betti_map_0':data_1['betti_map_list'][0],
                'betti_map_1': data_1['betti_map_list'][1],
                'betti_map_2': data_1['betti_map_list'][2],
                'betti_map_3': data_1['betti_map_list'][3],
                'betti_map_4': data_1['betti_map_list'][4],
                 'weight_patch_onehot': data_1['weight_patch_onehot'],
                }

        if self.post:
            result.update({
                'weight_grad':  data_1['weight_grad'],
                'weight_grad_smooth': filter(smoothing_2d(data_1['weight_grad'], sigma=15, kernel_size=51), thre=0.05),
                'softmax_coarse':  data_1['softmax_coarse'],
            })

        return result


    def get_sample(self, item):
        return self.get_data_dict_by_idx(item)

class ImageSegmentation_euler(Dataset):
    """Dataset for image segmentation."""

    def __init__(self, args, softmax_data, seg_data, csv_file_betti=None, patch_size_list=None, stride_size=None,):


        self.seg_data = seg_data
        self.softmax_data = softmax_data

        self.samples = []
        self.args = args

        self.betti_file_dir = csv_file_betti
        if self.betti_file_dir is not None and not os.path.exists(self.betti_file_dir):
            os.mkdir(self.betti_file_dir)       # input a betti path but doesn't exist the path
        self.patch_size_list = patch_size_list
        self.stride_size = stride_size


        data_list = os.listdir(self.softmax_data)
        data_list = [x for x in data_list if x.endswith('jpg')]

        for idx, img_name in enumerate(data_list):
            if True:
                softmax_path = os.path.join(self.softmax_data, img_name)
                img_base_name = img_name.split('.')[0]
                seg_name = img_base_name + '_boundary.png'              # specific for cremi dataset
                seg_path = os.path.join(self.seg_data, seg_name)        # 'gif'

                assert os.path.exists(seg_path), 'No segmentation data in ' + seg_path


                sample = {
                          'labelmap': seg_path,
                          'softmax': softmax_path,
                          }

                if self.betti_file_dir is not None:
                    betti_path = os.path.join(self.betti_file_dir, img_base_name + '_betti.pkl')
                    sample.update({'betti': betti_path})
                else:
                    sample.update({'betti': None})
                self.samples.append(sample)
            else:
                break


    def __len__(self):
        return len(self.samples)

    def read_onehot_labelmap(self, file_path):
        labelmap_onehot = sitk.ReadImage(file_path, sitk.sitkVectorFloat32)
        labelmap_sitk = labelmap_onehot
        labelmap_onehot = torch.from_numpy(sitk.GetArrayFromImage(labelmap_onehot)).permute(3, 0, 1,2)
        return labelmap_onehot, labelmap_sitk


    def calculate_gt_betti(self, gt, patch_size_list=[8, 16, 32, 64, 128], stride_size=8):
        betti, _ = get_topo_gudhi(gt)          #gt.shape should be 2d???
        betti_map_list = []
        for i, patch_size in enumerate(patch_size_list):
            patch_no = int((gt.shape[-1] - patch_size) / stride_size + 1)
            betti_map = np.zeros((2, patch_no, patch_no))       # betti 0 and betti 1

            for x_idx in range(patch_no):
                for y_idx in range(patch_no):
                    xmin = x_idx * stride_size
                    ymin = y_idx * stride_size
                    xmax = xmin + patch_size
                    ymax = ymin + patch_size
                    gt_patch = gt[xmin: xmax, ymin: ymax]
                    betti_pred, _ = get_topo_gudhi(gt_patch)
                    betti_map[0, x_idx, y_idx] = betti_pred[0]
                    betti_map[1, x_idx, y_idx] = betti_pred[1]

            betti_map_list.append(betti_map)

        betti_dict = {'betti_number': betti,
                      'betti_map_list': betti_map_list,
                      }

        return betti_dict

    def zscore(self, image):
        mean = image.mean()
        std = image.std()
        out = (image - mean) / std
        return out, mean, std

    def zeroone(self, image):
        mmax = image.max()
        mmin = image.min()
        out = (image - mmin)/ (mmax - mmin+ 0.001)
        return out

    def get_seg(self, seg_path, w=None):
        labelmap = cv2.imread(seg_path)
        labelmap = (np.array(labelmap[..., 0]) / 255)
        if w is not None:
            pass
        else:
            w = [0.1, 0.9]
        weight = torch.zeros(labelmap.shape)
        for i in range(labelmap.shape[0]):
            for j in range(labelmap.shape[1]):
                weight[i, j] = w[int(labelmap[i, j])]
        return labelmap, weight


    def get_img_zero_one(self, img_path):
        # read an image, and map it to [0-1]. e.g. weight_grad and softmax map
        img = cv2.imread(img_path)[..., 0:1]
        img = np.array(img).astype(np.float32)
        img_norm = self.zeroone(img)
        return img_norm


    def get_data_dict_by_dir(self, softmax_path, seg_path, betti_path=None, ):
        ''' save all the image, labelmap data into one pkl'''
        img_fname = os.path.basename(softmax_path)
        seg_pkl_path = seg_path[:-4] + '.pkl'

        softmax_coarse_pkl_path = softmax_path[:-4] + '.pkl'
        if os.path.exists(softmax_coarse_pkl_path):
            with open(softmax_coarse_pkl_path, 'rb') as f:
                softmax_coarse = pickle.load(f)
        else:
            softmax_coarse = self.get_img_zero_one(softmax_path)
            with open(softmax_coarse_pkl_path, 'wb') as f:
                pickle.dump(softmax_coarse, f)

        if os.path.exists(seg_pkl_path):
            with open(seg_pkl_path, 'rb') as f:
                seg_dict = pickle.load(f)
            labelmap = seg_dict['labelmap']
            weight = 2 * seg_dict['weight']
        else:
            labelmap, weight = self.get_seg(seg_path, self.weight)
            with open(seg_pkl_path, 'wb') as f:
                pickle.dump({
                    'labelmap': labelmap,
                    'weight': weight,
                }, f)

        labelmap_argmax = torch.from_numpy(labelmap).type(torch.float32)  # .unsqueeze(0)
        labelmap_onehot = labelmap_argmax.unsqueeze(dim=0)
        softmax_coarse = torch.from_numpy(softmax_coarse).type(torch.float32).unsqueeze(dim=0)


        result = {'img_fname': img_fname,
                  'labelmap_argmax': labelmap_argmax,
                  'labelmap_onehot': labelmap_onehot,
                  'softmax_coarse': softmax_coarse,
                  }

        '''process betti data'''
        t3 = time.time()
        if betti_path is not None:      #  either read or write
            if os.path.exists(betti_path):
                with open(betti_path, 'rb') as f:
                    betti_dict = pickle.load(f)
                # print('betti_number in dataset.py, load', betti_dict['betti_number'].dtype)
            else:
                betti_dict = self.calculate_gt_betti(labelmap, self.patch_size_list, self.stride_size)
                with open(betti_path, 'wb') as f:
                    pickle.dump(betti_dict, f)
                print('done', betti_path)
            result.update({'betti_number': betti_dict['betti_number'],
                           'betti_map_list': betti_dict['betti_map_list'],
                           })
        # print(time.time()-t3)
        return result

    def get_data_dict_by_idx(self, item):
        sample = self.samples[item]
        img_path = sample['image']
        seg_path = sample['labelmap']
        softmax_path = sample['softmax']
        weight_grad_path = sample['weight_grad']
        # mask_path = sample['mask']

        result = self.get_data_dict_by_dir(img_path, seg_path, softmax_path=softmax_path, weight_grad_path=weight_grad_path)  #, mask_path)
        return result


    def __getitem__(self, item):
        # print('item:', item)

        # return sample
        """self.sample_index_according_to_age 是所有数据的index按照age的排序"""
        # print('item', item)
        sample = self.samples[item]
        data_1 = self.get_data_dict_by_dir(sample['softmax'], sample['labelmap'], sample['betti'],
                                           )

        result= {
                'softmax_coarse':  data_1['softmax_coarse'],
                'labelmap_argmax': data_1['labelmap_argmax'],
                'labelmap_onehot': data_1['labelmap_onehot'],
                'fname': data_1['img_fname'],
                'betti_number': data_1['betti_number'],
                'betti_map_0':data_1['betti_map_list'][0],
                'betti_map_1': data_1['betti_map_list'][1],
                'betti_map_2': data_1['betti_map_list'][2],
                'betti_map_3': data_1['betti_map_list'][3],
                'betti_map_4': data_1['betti_map_list'][4],
            }

        return result


    def get_sample(self, item):
        return self.get_data_dict_by_idx(item)

