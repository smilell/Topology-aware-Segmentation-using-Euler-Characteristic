import os
import argparse
import sys

sys.path.append('../../../')
from train import train, test_single_stage, euler, test_two_stages

import matplotlib as mpl
from utils.utils import mk_dirs
back_end = mpl.get_backend()


separator = '----------------------------------------'

if __name__ == '__main__':
    # Set up argument parser
    #-----------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Topology-aware Segmentatstrion')
    parser.add_argument('--dev_true', default='0', type=str, help='cuda device (default: 0)')
    parser.add_argument('--dev', default='0', type=str, help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset', default='cremi', choices=['cremi',], help='training image path')
    parser.add_argument('--train', default='./data/cremi_2d/train/image/', help='training image path')
    parser.add_argument('--train_seg', default='./data/cremi_2d/train/label/', help='training labelmap path')
    parser.add_argument('--val', default='./data/cremi_2d/test/image/', help='validation image path')
    parser.add_argument('--val_seg', default='./data/cremi_2d/test/label/', help='validation labelmap path')
    parser.add_argument('--test', default='./data/cremi_2d/test/image/', help='test image path')
    parser.add_argument('--test_seg', default='./data/cremi_2d/test/label/', help='test labelmap path')

    parser.add_argument('--train_betti', default='./data/cremi_2d/train/betti/',
                        help='Directory to save topology metrics (betti number and euler characteristic) for training data. '
                             'If there is no files in this directory, '
                             'it will be automatically generated and saved when initialize the dataloader. '
                             'It could take time.')
    parser.add_argument('--test_betti', default='./data/cremi_2d/test/betti/',
                        help='directory to save topology metrics for test data.')
    parser.add_argument('--val_betti', default='./data/cremi_2d/test/betti/',
                        help='directory to save topology metrics for test data. ')

    # for training the second segmentation network TFS
    parser.add_argument('--train_ec', default=None, help='training image path')         #'./data/cremi_2d/train/ec/'
    parser.add_argument('--train_softmax', default=None, help='training labelmap path')     # './data/cremi_2d/train/softmax/'
    parser.add_argument('--val_ec', default=None, help='validation image path')     # './data/cremi_2d/test/ec/'
    parser.add_argument('--val_softmax', default=None, help='validation labelmap path')     # './data/cremi_2d/test/softmax/'
    parser.add_argument('--test_ec', default=None, help='test image path')      # './data/cremi_2d/test/ec/'
    parser.add_argument('--test_softmax', default=None, help='test labelmap path')      #'./data/cremi_2d/test/softmax/'

    parser.add_argument('--mode3d', default=False, action='store_true', help='3d dataset or not. ')

    # -------only for plug-in-and-play euler visualization---------------------------------------------------------------------------------
    parser.add_argument('--ec_pred', default=None, help='ec input')
    parser.add_argument('--ec_gt', default=None, help='ec input')
    #-----------------------------------------------------------------------------------------------------------------
    parser.add_argument('--phase', default='test_pre', type=str,
                        choices=['train_pre', 'train_post', 'test', 'test_pre', 'test_post', 'euler_visualization', ])

    parser.add_argument('--visual_euler', default=False, action='store_true',
                        help='Calculate and visualize Euler characteristic map during training')
    parser.add_argument('--trained_model_pre', default=None, help='The directory to load the trained stage 1 model')
    parser.add_argument('--trained_model_pre_type', default='best', choices=['best', 'latest'], help='Load stage 1 model, if it is the best performance on validation set (usually for test), or the latest epoch (to continue training)')
    parser.add_argument('--trained_model_post', default=None, help='The directory to load the trained stage 2 model')
    parser.add_argument('--trained_model_post_type', default='best', choices=['best', 'latest'], help='Load stage 2 model')

    parser.add_argument('--model', default=None, help='The directory to save trained model parameters')         # './model/EC_1/'
    parser.add_argument('--TBout', default=None, help='The directory to save tensorboard output files during both training and test')     # './output/EC_1/'
    parser.add_argument('--out', default=None,
                        help='The directory to save test results. 3 folders will be generated in this directory: '
                             'softmax, argmax and ec')
    parser.add_argument('--config', default='config.json', help='config file')

    #-----------------------------------------------------------------------------------------------------------------
    # parser.add_argument('--inverse', default=True, action='store_true', help='Calculate EC twice on the inverse map')  #
    parser.add_argument('--post_type', default=None,
                        choices=['mix_topo_gaussian', 'diff_gaussian', 'no_inpainting'],
                        help='some default inpainting settings,'
                             'mix_topo_gaussian (default): inpainting_region=topo, inpaint_type=gaussian, inpainting_threshold=r, inpaint_mix=True, inpaint_downsample=False' 
                             'diff_gaussian: inpainting_region=random, inpaint_type=gaussian, inpainting_threshold=0.5, inpaint_mix=True, inpaint_downsample=False'
                             'no_inpainting: inpainting_region=No, inpaint_type=gaussian, inpainting_threshold=1, inpaint_mix=False, inpaint_downsample=Fals' )
    parser.add_argument('--inpainting_region', default='topo', type=str, choices=['topo', 'diff', 'random','No'],
                        help='topo: using Euler error region'
                             'diff: using the difference map between argmax prediction and gt'
                             'random: randomly mask 5 blocks as the input'
                        )
    parser.add_argument('--inpaint_type', default='gaussian', type=str, choices=['gaussian', 'bw', 'gray'])
    parser.add_argument('--inpainting_threshold', default='r', type=str)        # default = '0.4'
    parser.add_argument('--inpaint_mix', default=True, help='if inpaint_mix, only have samples will be inpaint')
    parser.add_argument('--inpaint_downsample', default=None, help='downsample factor, could be 0.5')

    parser.add_argument('--loss_basic', default='ce', choices=['ce', 'dice_ce', 'contour', 'cldice', 'warp', 'ph', 'no', 'mse', 'ce_seg_cla'], help='loss to train the baseline segmentation network')
    parser.add_argument('--weightedLoss', default=False, action='store_true', help='add weights to cross enrtopy loss')
    parser.add_argument('--weightedLossMode', default=None, choices=['0', '1', '2', None], help='add weights to cross enrtopy loss')
    parser.add_argument('--loss_others_weight', default=1, help='add weights to cross enrtopy loss')

    parser.add_argument('--isBetti', default=False, action='store_true', help='calculate betti number and betti errorfor the predicted map.'
                                                                              'betti number for GT on different patches are saved in args.betti_train/test/validate')

    parser.add_argument('--isRandomPatchSizeForEC', default=False, action='store_true')
    parser.add_argument('--patch_size_list', default=[8, 16, 32, 64, 128], help='When initialize the dataset, pre-calculate/load patch-wise EC and betti map. Calculated based on gudhi package')
    parser.add_argument('--subpatch_size', default=16, type=int, help='Patch size to calculate EC map in TVD block, 16 for cow, 32 for cremi')
    parser.add_argument('--subpatch_stride', default=1, type=int, help='Patch stride to calculate EC map in TVD block')
    parser.add_argument('--kernel_size', default=7, type=int, help='smoothing parameters to calculate EC map in TVD block')
    parser.add_argument('--sigma', default=3, type=int, help='smoothing parameters to calculate EC map in TVD block')
    parser.add_argument('--thre', default=0.4, type=float, help='smoothing parameters to calculate EC map in TVD block')

    #-----------------------------------------------------------------------------------------------------------------
    parser.add_argument('--train_info', default=None, help='training image path')
    parser.add_argument('--val_info', default=None, help='validation image path')
    parser.add_argument('--test_info', default=None, help='test image path')

    args = parser.parse_args()

    print(type(args.dev), args.out)

    if args.phase == 'train_pre':
        print(separator)
        print('Start to train segmentation model stage 1, model and tensorboard files will be saved in {} and {}'.format(args.model, args.TBout))
        print(separator)
        mk_dirs(args.TBout, args.model)
        train(args)

        args.visual_euler = True
        args.isBetti = True
        args.trained_model_pre = args.model
        args.out = args.TBout
        args.phase = 'test_pre'
        print('Start to test stage 1 segmentation model {} , results will be saved in {}'.format(args.trained_model_pre, args.out))
        print(separator)
        mk_dirs(os.path.join(args.out))
        test_single_stage(args)

    elif args.phase == 'train_post':
        print(separator)
        print('Start to train segmentation model stage 2, model and tensorboard files will be saved in {} and {}'.format(args.model, args.TBout))
        print(separator)
        mk_dirs(args.TBout, args.model)
        train(args)

        args.visual_euler = True
        args.isBetti = True
        args.trained_model_post = args.model
        args.out = args.TBout
        args.phase = 'test_post'
        args.test_ec = args.val_ec
        args.test_softmax = args.val_softmax
        print('Start to test stage 2 segmentation model {}, results will be saved in {}'.format(args.trained_model_post, args.out))
        print(separator)
        mk_dirs(os.path.join(args.out))
        test_single_stage(args)

    elif args.phase == 'test':
        print(separator)
        print('Start to test both stage 1 and stage 2 segmentation models {} and {}, results will be saved in {}'.format(args.trained_model_pre, args.trained_model_post, args.out))
        print(separator)
        mk_dirs(args.out, args.TBout,)
        args.visual_euler = True
        test_two_stages(args)
    elif args.phase == 'test_pre':
        print(separator)
        print('Start to test stage 1 segmentation model {} , results will be saved in {}'.format(args.trained_model_pre, args.out))
        print(separator)
        mk_dirs(os.path.join(args.out))
        test_single_stage(args)
    elif args.phase == 'test_post':
        print(separator)
        print('Start to test stage 2 segmentation model {}, results will be saved in {}'.format(args.trained_model_post, args.out))
        print(separator)
        mk_dirs(os.path.join(args.out))
        test_single_stage(args)
    elif args.phase == 'euler_visualization':
        print(separator)
        print('Start to generate euler error map, results will be saved in {}'.format(args.out))
        print(separator)
        args.visual_euler = True
        mk_dirs(os.path.join(args.out))
        euler(args)
