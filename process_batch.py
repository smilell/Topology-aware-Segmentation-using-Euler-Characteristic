import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from loss import DiceLoss, softDiceLoss, topo_loss, topo_loss_2d, contour_loss, warping_loss
from utils.utils import *
from utils.topo import get_topo_gudhi

import utils.metrics as metrics
import torchvision
import numbers

def norm1(x, mmax=None, mmin=None):
    x_norm = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_slice = x[i,...]
        if mmax is None:
            mmax = x_slice.max()
        if mmin is None:
            mmin= x_slice.min()
        x_norm[i, ...] = (x_slice - mmin) / (mmax - mmin + 0.001)

    return x_norm

def clip(x, filter_per=0.005):
    x_norm = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_slice = x[i,...]

        topk = torch.topk(x_slice.reshape(-1), int(filter_per * x_slice.shape[1] * x_slice.shape[2] * x_slice.shape[0]))
        mmax = topk.values.min()
        x_slice[x_slice >= mmax] = mmax
        x_norm[i, ...] = x_slice
    return x_norm

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def smoothing_2d(x, kernel_size=21, sigma=3):
    blur = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    if len(x.shape) == 3:
        x = x.unsqueeze(dim=1)
        x_blur = blur(x).squeeze(dim=1)
        # plt.figure(figsize=(40, 20))
        # plt.subplot(1, 2, 1).imshow(x[0,0,...].cpu().detach().numpy())
        # plt.subplot(1, 2, 2).imshow(x_blur[0,...].cpu().detach().numpy())
        # plt.show()
    else:
        x_blur = blur(x)
    return x_blur

def smoothing_3d(x,  kernel_size=21, sigma=3):
    b, c, _, __, ___ = x.shape
    blur = GaussianSmoothing(channels=c, kernel_size=kernel_size, sigma=sigma, dim=3).to(x.device)
    pad_size = kernel_size // 2
    x_pad = F.pad(x, (pad_size, pad_size, pad_size, pad_size,pad_size,pad_size), mode='replicate')          # should be reflect, but pytorch >=1.9 support ReflectionPad3d...
    # pad = nn.ReflectionPad3d((2, 2, 2, 2, 2, 2))
    # x = pad(x)
    x_blur = blur(x_pad)
    return x_blur

def remap(x, mmax=0.1, mmin=0.9):
    x_norm = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_slice = x[i,...]
        x_max = x_slice.max()
        x_min = x_slice.min()
        # print(x_max, x_min)
        x_norm[i, ...] = (x_slice - x_min) / (x_max - x_min + 0.001)

    x_re = (mmax-mmin) * x_norm + mmin
    return x_re

def inpainting(pred_softmax, weight_grad, threshold=0.6, ttype='gaussian', downsample=None):
    image_inpainted = torch.clone(pred_softmax).detach()
    weight_grad_1 = torch.clone(weight_grad).detach()
    weight_grad_1[weight_grad_1 >= threshold] = 1
    weight_grad_1[weight_grad_1 < threshold] = 0

    if downsample is not None:
        weight_grad_1 = torch.nn.functional.interpolate(weight_grad_1, size=None, scale_factor=0.5, mode='bilinear', )
        # weight_grad_1 = torch.nn.functional.interpolate(weight_grad_1, size=None, scale_factor=2, mode='bilinear', )
        weight_grad_1[weight_grad_1>=0.5] = 1
        weight_grad_1[weight_grad_1 < 0.5] = 0
    else:
        pass

    if ttype == 'gaussian':
        noise = torch.randn_like(weight_grad_1)
        c = torch.nn.Sigmoid()
        noise_sigmoid = c(noise)
    elif ttype == 'bw':
        if random.randint(0,1):
            noise_sigmoid = torch.zeros_like(weight_grad_1)
        else:
            noise_sigmoid = torch.ones_like(weight_grad_1)
    elif ttype == 'gray':
        noise_sigmoid = torch.ones_like(weight_grad_1) * 0.5
    image_inpainted[weight_grad_1 ==1] = noise_sigmoid[weight_grad_1 ==1]

    return image_inpainted, weight_grad_1

def inpaint_img(args, pred_softmax, weight_grad, bbox_size=32, bbox_num=5):
    if args.post_type == 'mix_topo_gaussian':
        args.inpainting_region, args.inpaint_type, args.inpainting_threshold, args.inpaint_mix = ['topo', 'gaussian', 'r', True]
    elif args.post_type == 'mix_random':
        args.inpainting_region, args.inpaint_type, args.inpainting_threshold, args.inpaint_mix = ['random', 'gaussian', 0.5, True]
    elif args.post_type == 'no_inpainting':
        args.inpainting_region, args.inpaint_type, args.inpainting_threshold, args.inpaint_mix = ['No', 'gaussian', 1, False]

    # ____________________________________________________________________________________________________________
    if args.inpainting_region == 'topo':
        inpaint_map = weight_grad
    elif args.inpainting_region == 'diff':
        inpaint_map = (labelmap_argmax - pred_argmax).abs().unsqueeze(dim=1)
    elif args.inpainting_region == 'random':
        inpaint_map = torch.zeros_like(weight_grad)
        bbox = bbox_size
        shape_x = inpaint_map.shape[2]
        shape_y = inpaint_map.shape[3]
        for i in range(bbox_num):
            bbox_start_x = random.randint(0, shape_x - bbox - 1)
            bbox_start_y = random.randint(0, shape_y - bbox - 1)
            inpaint_map[:, :, bbox_start_x:bbox_start_x + bbox, bbox_start_y:bbox_start_y + bbox] = 1
    elif args.inpainting_region == 'No':
        inpaint_map = torch.zeros_like(weight_grad)
    # ____________________________________________________________________________________________________________
    inpaint_type = args.inpaint_type
    # ____________________________________________________________________________________________________________
    if args.inpainting_threshold == 'r':
        inpainting_threshold = float(random.randint(1, 9) / 10)
    else:
        inpainting_threshold = float(args.inpainting_threshold)

    # ____________________________________________________________________________________________________________
    downsample_factor = args.inpaint_downsample
    # ____________________________________________________________________________________________________________
    if args.inpaint_mix and random.randint(0, 1):
        pred_inpainted = pred_softmax
    else:
        pred_inpainted, weight_map = inpainting(pred_softmax, inpaint_map, threshold=inpainting_threshold, ttype=inpaint_type, downsample=downsample_factor)


    return pred_inpainted

def getGradMap(loss, ec_net, map):
    ec_net.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)            # don't delete the graph after this backward. still save the graph within each batch
    gradients = map.grad
    ec_net.zero_grad(set_to_none=True)
    return gradients

def TVD(args, euler_input_pred, euler_input_gt, ec_net, isBinary=False, thre=1.5, kernel_size=11, sigma=3):
    euler_input_pred = euler_input_pred.detach()
    euler_input_gt = euler_input_gt.detach()
    euler_input_pred.requires_grad = True
    euler_input_gt.requires_grad = True

    euler_dict_pred = ec_net(euler_input_pred, isReturnEcMap=True, isReturnInternalFeatures=True)
    EC_pred_patch = euler_dict_pred['EC_map']

    euler_dict_gt = ec_net(euler_input_gt, isReturnEcMap=True, isReturnInternalFeatures=True)
    EC_gt_patch = euler_dict_gt['EC_map']

    # EC_error_map = torch.abs(EC_pred_patch - EC_gt_patch)


    weight_grad_pred = getGradMap(F.mse_loss(EC_pred_patch, EC_gt_patch.detach()), ec_net, euler_input_pred)
    weight_grad_gt = getGradMap(F.mse_loss(EC_gt_patch, EC_pred_patch.detach()), ec_net, euler_input_gt)

    weight_grad_pred = clip(weight_grad_pred.abs(), filter_per=0.005)
    weight_grad_gt = clip(weight_grad_gt.abs(), filter_per=0.005)

    weight_grad_pred = norm1(weight_grad_pred)  # , mmax=0.0067, mmin=0)
    weight_grad_gt = norm1(weight_grad_gt)  # , mmax=0.0067, mmin=0)

    weight_grad = torch.where(weight_grad_pred > weight_grad_gt, weight_grad_pred, weight_grad_gt)
    if args.mode3d:
        weight_grad = smoothing_3d(weight_grad, kernel_size=int(kernel_size), sigma=int(sigma))
    else:
        weight_grad = smoothing_2d(weight_grad, kernel_size=int(kernel_size), sigma=int(sigma))

    weight_grad = remap(weight_grad, mmax=2, mmin=0)
    if isBinary:
        weight_grad[weight_grad >= float(thre)] = 2
        weight_grad[weight_grad < float(thre)] = 0

    return weight_grad, weight_grad_pred, weight_grad_gt#, EC_error_map.unsqueeze(dim=1)



def process_batch_opti_process(args, config, batch_samples):
    # ------------------------------ 0. decide run pre model / post model ------------------------------
    if args.phase in ['train_pre', 'test_pre', 'train_patch', 'test_patch']:
        model_type = 'PRE'
    elif args.phase in ['train_post', 'test_post', 'train_patch_post', 'test_post_multipile']:
        model_type = 'POST'
    elif args.phase in ['train_patch_post_multitask', 'train_patch_post_multitask_duplicate_decoder']:
        model_type = 'MULTI'
    else:
        raise 'Calling wrong process_batch code. Check if it should be process_batch_two_stages() if test end-to-end or process_batch_ec_only() if only plug-and-play TVD block'
    # ------------------------------ 1. fetch data ------------------------------
    image = batch_samples['image'].to(config.device)        # normalized
    image_ori = batch_samples['image_ori'].to(config.device)        # just for tensorboard visualization, before normalization

    labelmap_argmax = batch_samples['labelmap_argmax'].to(config.device)       # [batch, w, h]
    labelmap_onehot = batch_samples['labelmap_onehot'].to(config.device)       # [batch, c, w, h]

    # ------------------------------ 2. forward ------------------------------
    c = torch.nn.Sigmoid()

    if model_type == 'PRE':
        pred = config.model(image)
        pred_softmax = c(pred)
    elif model_type == 'POST':
        pred_softmax_coarse = batch_samples['softmax_coarse'].to(config.device)
        if config.model.training:
            weight_grad = batch_samples['weight_grad'].to(config.device)
            # pred_argmax_coarse = (pred_softmax_coarse >= 0.5).type(torch.float32).to(config.device).squeeze(dim=1)
            pred_softmax_coarse_inpainted = inpaint_img(args, pred_softmax_coarse, weight_grad)
        else:
            pred_softmax_coarse_inpainted = pred_softmax_coarse

        pred = config.model(pred_softmax_coarse_inpainted)
        pred_softmax = c(pred)
    elif model_type in ['MULTI']:
        pred_softmax_coarse = batch_samples['softmax_coarse'].to(config.device)
        if config.model.training:
            weight_grad = batch_samples['weight_grad'].to(config.device)
            # pred_argmax_coarse = (pred_softmax_coarse >= 0.5).type(torch.float32).to(config.device).squeeze(dim=1)
            pred_softmax_coarse_inpainted = inpaint_img(args, pred_softmax_coarse, weight_grad)

        else:
            pred_softmax_coarse_inpainted = pred_softmax_coarse
        pred, pred_clas = config.model(pred_softmax_coarse_inpainted)
        pred_clas_softmax = c(pred_clas)
        pred_softmax = c(pred)


    if config.config['num_classes'] == 1:       # config.config['num_classes'] it only counts foreground
        pred_argmax = (pred_softmax>=0.5).type(torch.float32).to(config.device).squeeze(dim=1)
        pred_onehot = pred_argmax.unsqueeze(dim=1)
    else:
        pred_argmax = torch.argmax(pred_softmax, dim=1)
        pred_onehot = torch.nn.functional.one_hot(pred_argmax, num_classes=config.config['num_classes'] + 1).permute(0, -1, 1, 2, 3)

    # ------------------------------ 3. evaluation ------------------------------
    dice = metrics.dice_score(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes'] + 1)

    # dice = metrics.dice_score(pred_onehot, labelmap_onehot, num_classes=config.config['num_classes'] + 1, one_hot=True)
    ## further evaluations. Hausdoff and ASD could take time
    # precision = metrics.precision(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes']+1)
    # recall = metrics.recall(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes']+1)
    # specificity = metrics.specificity(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes']+1)

    # if pred_argmax.sum() == 0 or labelmap_argmax.sum() == 0:
    #     if labelmap_argmax.sum() == 0:
    #         print('hold on', batch_samples['fname'], 'all_zero gt')
    #     else:
    #         print('hold on', batch_samples['fname'], 'all_zero prediction')
    #     hausdoff = np.zeros(dice.shape)
    #     asd = np.zeros(dice.shape)
    # else:
    #     hausdoff = metrics.hausdorff_distance(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes'] + 1)
    #     asd = metrics.average_surface_distance(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes'] + 1)

    if args.isBetti:         # Topo evaluation
        # 1. get gt betti info
        betti_gt = batch_samples['betti_number']
        # 2. get pred betti info
        betti_pred = np.zeros((pred_argmax.shape[0], 3))
        for ii in range(pred_argmax.shape[0]):
            betti_pred[ii, ...], info_2 = get_topo_gudhi(
                pred_argmax[ii, ...].cpu().detach().numpy())

        betti_error = np.abs(betti_pred - betti_gt.numpy())
        betti_error = betti_error.mean(axis=0)
        betti_pred = betti_pred.mean(axis=0)
        betti_gt = betti_gt.mean(axis=0)

    # ------------------------------ 4. Loss ------------------------------
    if args.loss_basic in ['ce', 'dice_ce', 'contour', 'cldice', 'warp', 'ph', 'ce_seg_cla']:
        if config.config['num_classes'] == 1:
            if args.weightedLoss and config.model.training:
                weight_patch_onehot = batch_samples['weight_grad_smooth'].to(config.device)
                loss_crossEntropy = F.binary_cross_entropy(pred_softmax, labelmap_onehot, weight=weight_patch_onehot)
            else:
                loss_crossEntropy = F.binary_cross_entropy(pred_softmax, labelmap_onehot)


        else:
            assert False, 'Only support binary segmentation now.'

        if args.loss_basic == 'ce':
            loss_others = torch.zeros_like(loss_crossEntropy)

        elif args.loss_basic == 'contour':
            loss_others = contour_loss(labelmap_onehot, pred_softmax)

        elif args.loss_basic == 'cldice':
            cl_dice = soft_dice_cldice()
            loss_others = cl_dice(labelmap_onehot, pred_softmax)

        elif args.loss_basic == 'dice_ce':
            dice_loss = DiceLoss()
            loss_others = dice_loss(pred_softmax, labelmap_onehot)

        elif args.loss_basic == 'warp':
            loss_warp, critical_pt_no = warping_loss(pred_softmax, labelmap_onehot, )
            loss_others = 0.0001 * loss_warp

        elif args.loss_basic == 'ph':
            if args.mode3d:
                Topo_loss = topo_loss(package='gudhi')
            else:
                Topo_loss = topo_loss_2d(package='gudhi')  # or cripser
            loss_topo_all, betti_pred_all, num_points_updating, topo_weight_map, topo_ref_map = Topo_loss(pred_softmax, labelmap_onehot, config.device)
            loss_others = 0.001 * loss_topo_all[0]
        elif args.loss_basic == 'ce_seg_cla':
            # print(batch_samples['category_idx_onehot'].shape, batch_samples['category_idx_onehot'])
            # print(pred_clas.shape)
            label_clas_onehot = batch_samples['category_idx_onehot'].to(config.device)
            label_clas = label_clas_onehot.squeeze(dim=1)

            loss_others = F.binary_cross_entropy(pred_clas_softmax, label_clas_onehot)          # shape = [4, 1]
            pred_clas_argmax = torch.zeros_like(pred_clas_softmax)
            pred_clas_argmax[pred_clas_softmax>=0.5] = 1
            pred_clas_argmax = pred_clas_argmax.squeeze(dim=1)
            # good label = 0, bad label = 1
            bad_no = label_clas.sum()
            good_no = len(label_clas) - bad_no
            acc_clas = (label_clas == pred_clas_argmax).sum() / (len(label_clas)+ 0.000001)
            acc_clas_good = ((1-label_clas) * (1-pred_clas_argmax)).sum() / (good_no+ 0.000001)
            acc_clas_bad = (label_clas * pred_clas_argmax).sum() / (bad_no+ 0.000001)

    elif args.loss_basic == 'no':
        loss_crossEntropy = torch.from_numpy(np.zeros(1)).to(config.device)
        loss_others = torch.zeros_like(loss_crossEntropy)
    else:
        assert False, 'Not support other kind of weighted loss. Not recognizable loss function'
    loss_others =  float(args.loss_others_weight) * loss_others
    loss = loss_crossEntropy + loss_others

    # ------------------------------ 5. Euler visualization ------------------------------
    if args.visual_euler:
        if config.config['num_classes'] == 1:
            euler_input_pred = pred_argmax.unsqueeze(dim=1).detach()
            euler_input_gt = labelmap_onehot.detach()
        else:
            # choose the cgm dim to calculate topo error
            cgm_dim = config.config['cgm_dim'][0]
            euler_input_pred = pred_argmax.type(torch.float32).detach()
            euler_input_pred[euler_input_pred != cgm_dim] = 0
            euler_input_pred[euler_input_pred == cgm_dim] = 1

            euler_input_gt = labelmap_onehot[:, cgm_dim:cgm_dim + 1, ...].detach()

        ec_error_map, ec_error_map_pred, ec_error_map_gt = TVD(args, euler_input_pred, euler_input_gt, ec_net=config.ec, isBinary=False, thre=args.thre, kernel_size=args.kernel_size, sigma=args.sigma)

    values_dict = {
                   '01_loss_all': loss.item(),
                   '02_loss_crossEntropy': loss_crossEntropy.item(),
                   '03_loss_others': loss_others.item(),
                   '04_dice': dice[1],#dice[1::].tolist(),
                    # '04_hausdoff': hausdoff[1::].tolist(),
                    # '05_asd': asd[1::].tolist(),
                   }
    if model_type == 'MULTI':
        values_dict.update({
            '05_acc_clas': acc_clas.item(),
            '06_acc_clas_good': acc_clas_good.item(),
            '07_acc_clas_bad': acc_clas_bad.item(),
        })

    if args.isBetti:
        values_dict.update({'10_betti_gt_0': betti_gt[0],
                            '11_betti_gt_1': betti_gt[1],
                            '12_bett_pred_0': betti_pred[0],
                            '13_bett_pred_1': betti_pred[1],
                            '14_bett_error_0': betti_error[0],
                            '15_bett_error_1': betti_error[1],
                            })

    # to visualize an image, using plot_3d_tensor(image_ori[0,...], color='gray')    # color='gray'
    images_dict = {
                    'image': image_ori,
                   'labelmap': labelmap_argmax.unsqueeze(dim=1),
                   'softmax': pred_softmax,
                   'argmax': pred_argmax.unsqueeze(dim=1),
                   }

    if args.weightedLoss and config.model.training:
        images_dict.update({
            'weight': weight_patch_onehot,
        })

    if model_type == 'POST':
        images_dict.update({
            'inpainted': pred_softmax_coarse_inpainted
        })

    if args.visual_euler:
        images_dict.update({
                            'ec': ec_error_map,
                            })
    # if True:
    #     patch_idxs = batch_samples['patch_idxs']
    #     image_whole = union(patch_idxs, image_ori, argmax=False)
    #     pred_softmax_whole = union(patch_idxs, pred_softmax, argmax=False)
    #     pred_argmax_whole = (pred_softmax_whole>=0.5).type(torch.float32).to(config.device).squeeze(dim=1)#union(patch_idxs, pred_argmax, argmax=True)
    #
    #     labelmap_argmax_whole =union(patch_idxs, labelmap_argmax, argmax=True)


    return loss, images_dict, values_dict

def process_batch_euler(args, config, batch_samples):

    assert args.phase == 'euler_visualization', 'wrong phase'

    # ------------------------------ 1. fetch data ------------------------------
    labelmap_argmax = batch_samples['labelmap_argmax'].to(config.device)       # [batch, w, h]
    labelmap_onehot = batch_samples['labelmap_onehot'].to(config.device)       # [batch, c, w, h]
    pred_softmax = batch_samples['softmax_coarse'].to(config.device)



    if config.config['num_classes'] == 1:       # config.config['num_classes'] it only counts foreground
        pred_argmax = (pred_softmax>=0.5).type(torch.float32).to(config.device).squeeze(dim=1)
        pred_onehot = pred_argmax.unsqueeze(dim=1)
    else:
        pred_argmax = torch.argmax(pred_softmax, dim=1)
        pred_onehot = torch.nn.functional.one_hot(pred_argmax, num_classes=config.config['num_classes'] + 1).permute(0, -1, 1, 2, 3)

    # ------------------------------ 3. evaluation ------------------------------
    dice = metrics.dice_score(pred_argmax, labelmap_argmax, num_classes=config.config['num_classes'] + 1)


    if args.isBetti:         # Topo evaluation
        # 1. get gt betti info
        betti_gt = batch_samples['betti_number']
        # 2. get pred betti info
        betti_pred = np.zeros((pred_argmax.shape[0], 3))
        for ii in range(pred_argmax.shape[0]):
            betti_pred[ii, ...], info_2 = get_topo_gudhi(
                pred_argmax[ii, ...].cpu().detach().numpy())

        betti_error = np.abs(betti_pred - betti_gt.numpy())
        betti_error = betti_error.mean(axis=0)
        betti_pred = betti_pred.mean(axis=0)
        betti_gt = betti_gt.mean(axis=0)


    # ------------------------------ 5. Euler visualization ------------------------------
    if args.visual_euler:
        if config.config['num_classes'] == 1:
            euler_input_pred = pred_argmax.unsqueeze(dim=1).detach()
            euler_input_gt = labelmap_onehot.detach()
        else:
            # choose the cgm dim to calculate topo error
            cgm_dim = config.config['cgm_dim'][0]
            euler_input_pred = pred_argmax.type(torch.float32).detach()
            euler_input_pred[euler_input_pred != cgm_dim] = 0
            euler_input_pred[euler_input_pred == cgm_dim] = 1

            euler_input_gt = labelmap_onehot[:, cgm_dim:cgm_dim + 1, ...].detach()

        ec_error_map, ec_error_map_pred, ec_error_map_gt = TVD(args, euler_input_pred, euler_input_gt, ec_net=config.ec)

    values_dict = {
                   '04_dice': dice[0],
                    # '04_hausdoff': hausdoff[1::].tolist(),
                    # '05_asd': asd[1::].tolist(),
                   }

    if args.isBetti:
        values_dict.update({'10_betti_gt_0': betti_gt[0],
                            '11_betti_gt_1': betti_gt[1],
                            '12_bett_pred_0': betti_pred[0],
                            '13_bett_pred_1': betti_pred[1],
                            '14_bett_error_0': betti_error[0],
                            '15_bett_error_1': betti_error[1],

                            })

    # to visualize an image, using plot_3d_tensor(image_ori[0,...], color='gray')    # color='gray'
    images_dict = {
                   'labelmap': labelmap_argmax.unsqueeze(dim=1),
                   'softmax': pred_softmax,
                   'argmax': pred_argmax.unsqueeze(dim=1),
                   }


    if args.visual_euler:
        images_dict.update({
                            'ec': ec_error_map,
                            })

    return images_dict, values_dict

