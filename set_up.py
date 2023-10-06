import json
import os
import torch
import sys
sys.path.append('../../../')

from network import *
from utils.processing import zero_mean_unit_var, range_matching, zero_one, threshold_zero
from utils.transforms import Resampler, Normalizer

from attrdict import AttrDict

separator = '----------------------------------------'

def set_up_model(args):

    with open(args.config) as f:
        config = json.load(f)

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev_true
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(args.dev))))

    if config['normalizer'] == 'zero_mean_unit_var':
        normalizer = Normalizer(zero_mean_unit_var)
    elif config['normalizer'] == 'range_matching':
        normalizer = Normalizer(range_matching)
    elif config['normalizer'] == 'zero_one':
        normalizer = Normalizer(zero_one)
    elif config['normalizer'] == 'threshold_zero':
        normalizer = Normalizer(threshold_zero)
    elif config['normalizer'] == 'none':
        normalizer = None
    else:
        raise NotImplementedError('Normalizer {} not supported'.format(config['normalizer']))


    num_foreground = config['num_classes']
    if num_foreground == 1:
        num_channel_output = num_foreground
    else:
        num_channel_output = num_foreground + 1

    gamma = 0.5 ** (1 / config['epoch_decay_steps'])

    config_dict = {'config': config,
                   'device': device,
                   'normalizer': normalizer,
                   }

    if args.phase in ['train_pre', 'test_pre', ]:
        if not args.mode3d:
            preNet = UNet2D(num_classes=num_channel_output).to(device)
        parameters_preNet = list(preNet.parameters())
        optimizer_preNet = torch.optim.Adam(parameters_preNet, lr=config['learning_rate'])
        scheduler_preNet = torch.optim.lr_scheduler.ExponentialLR(optimizer_preNet, gamma, last_epoch=-1)

        config_dict.update({
            'model': preNet,
            'optimizer': optimizer_preNet,
            'scheduler': scheduler_preNet,
        })

    elif args.phase in ['train_post', 'test_post',]:
        if not args.mode3d:
            postNet = UNet2D(num_classes=num_channel_output).to(device)
        parameters_postNet = list(postNet.parameters())
        optimizer_postNet = torch.optim.Adam(parameters_postNet, lr=config['learning_rate'])
        scheduler_postNet = torch.optim.lr_scheduler.ExponentialLR(optimizer_postNet, gamma, last_epoch=-1)

        config_dict.update({
            'model': postNet,
            'optimizer': optimizer_postNet,
            'scheduler': scheduler_postNet,
        })


    elif args.phase == 'test':
        if not args.mode3d:
            preNet = UNet2D(num_classes=num_channel_output).to(device)
            postNet = UNet2D(num_classes=num_channel_output).to(device)

        config_dict.update({
            'preNet': preNet,
            'postNet': postNet,
        })


    if args.phase == 'euler_visualization' or args.visual_euler:
        print('initializing EC network, ', str(args.subpatch_size), str(args.subpatch_stride))
        if not args.mode3d:
            subpatch_size = int(args.subpatch_size)
            subpatch_stride = int(args.subpatch_stride)
            ec = EC(subpatch_size=subpatch_size, subpatch_stride=subpatch_stride).to(device)

        config_dict.update({
            'ec': ec,

        })

    return AttrDict(config_dict)
