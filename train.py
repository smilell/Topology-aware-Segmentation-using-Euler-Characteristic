import os
import pickle

import torch
from tqdm import tqdm
import numpy as np
from set_up import  set_up_model
from process_batch import  process_batch_opti_process, process_batch_euler
from utils.utils import write_values, write_images, mk_dirs, get_lr
import time
from tensorboardX import SummaryWriter
import cv2
import SimpleITK as sitk
from dataset import ImageSegmentation_cremi, ImageSegmentation_euler
import utils.metrics as metrics
from utils.utils import plot_3d_tensor


separator = '----------------------------------------'

def norm(pred_ce):
    # pred_ce = np.abs(pred_ce)
    mmax = pred_ce.max()
    mmin= pred_ce.min()
    pred_ce = (pred_ce - mmin) / ((mmax - mmin) + 0.000001)
    return pred_ce


def load_model(model_path, model, optimizer=None, device=None):
    if model_path is None:      # no pre trained model to load
        return 1, 0
    elif os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        start_epoch = checkpoint['epoch']
        if 'dice_record' in checkpoint.keys():
            dice_record = checkpoint['dice_record']
        else:
            dice_record = 0
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        return start_epoch, dice_record
    else:
        raise 'cannot load model from {}'.format(model_path)


def save_images(args, images_dict, output_dir, sample_name, pkl=True, hide_list=['labelmap', 'image']):
    # the results will be saved in K folders, each folder is named by the key name of images_dict,
    for k, folder_name in enumerate(images_dict.keys()):
        if not folder_name in hide_list:
            save_dir = os.path.join(output_dir, folder_name)
            if not os.path.exists(save_dir):
                mk_dirs(save_dir)
            sample_name_basic = sample_name[:-4] #+ '.jpg'
            image_save_name_jpg = os.path.join(save_dir, sample_name_basic + '.jpg')
            image_save_name_pkl = os.path.join(save_dir, sample_name_basic + '.pkl')
            image_save_name_nif = os.path.join(save_dir, sample_name_basic + '.nii.gz')
            if not args.mode3d:
                image_save_pkl = images_dict[folder_name][0, 0, ...].cpu().detach().numpy()
                image_save_jpg = (norm(image_save_pkl) * 255 ).astype(np.uint8)
                cv2.imwrite(image_save_name_jpg, image_save_jpg)
            else:
                image_save_pkl = images_dict[folder_name][0, 0, ...].cpu().detach().numpy()
                image_sitk = sitk.GetImageFromArray(image_save_pkl)
                sitk.WriteImage(image_sitk, image_save_name_nif)
            if pkl:
                with open(image_save_name_pkl, 'wb') as f:
                    pickle.dump(image_save_pkl, f)


def train(args):
    ''' ---------------------- 0. Set up  ---------------------- '''
    config = set_up_model(args)
    global_step = 0
    start_epoch = 1
    flag_init_logger = True
    dice_record_0 = 0
    writer = SummaryWriter('{}/tensorboard'.format(args.TBout))



    if args.phase == 'train_pre':
        loading_model_path = os.path.join(args.trained_model_pre, 'model_' + args.trained_model_pre_type + '.pt') if args.trained_model_pre is not None else None
    elif args.phase == 'train_post':
        loading_model_path = os.path.join(args.trained_model_post, 'model_' + args.trained_model_post_type + '.pt') if args.trained_model_post is not None else None
    else:
        raise 'Wrong phase type'

    ''' ---------------------- 1. Loading pre-trained models ---------------------- '''
    start_epoch, dice_record_0 = load_model(loading_model_path, config.model, config.optimizer, config.device)

    ''' ---------------------- 2. Initializing datasets ---------------------- '''
    print(separator)
    print('Loading training data...')
    print(separator)
    patch_info_path = args.train_info if args.weightedLoss else None

    weight_mode = int(args.weightedLossMode) if args.weightedLossMode is not None else None
    if args.dataset == 'cremi':
        dataset_train = ImageSegmentation_cremi(args, args.train, args.train_seg,
                                                softmax_data=args.train_softmax,
                                                weight_grad=args.train_ec,
                                                csv_file_betti=args.train_betti,
                                                patch_size_list=args.patch_size_list,
                                                stride_size=8,
                                                crop=False,
                                                flip=False,
                                                weight=config.config['dist'],
                                                multi=1,
                                                patch_info_path=patch_info_path,
                                                mode=weight_mode,
                                                )

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.config['batch_size'], shuffle=True, num_workers=8, prefetch_factor=8)
    print('done')
    print(separator)
    if args.val is not None:
        print(separator)
        print('VALIDATION data...')
        print(separator)
        if args.dataset == 'cremi':
            dataset_val = ImageSegmentation_cremi(args, args.val, args.val_seg,
                                                  softmax_data=args.val_softmax,
                                                  weight_grad=args.val_ec,
                                                  csv_file_betti=args.val_betti,
                                                  patch_size_list=args.patch_size_list,
                                                  stride_size=8,
                                                  crop=False,
                                                  weight=config.config['dist'],)


        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8,
                                                     prefetch_factor=8)
        print('done')
        print(separator)

    ''' ---------------------- 3. Start training ---------------------- '''
    for epoch in range(start_epoch, config.config['epochs'] + 1):        #config.config['epochs']
        T0 = 0
        T1 = 0
        T2 = 0

        config.model.train()
        t2 = time.time()

        ''' ---------------------- 3.1 Training block ---------------------- '''
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_train, desc='Epoch {}'.format(epoch))):

            global_step += 1
            t0 = time.time()
            T0 += (t0 - t2)

            loss, images_dict, values_dict = process_batch_opti_process(args, config, batch_samples)

            if flag_init_logger:
                """only run during the first batch in the first epoch"""
                loss_names = values_dict.keys()
                train_logger = metrics.Logger('TRAIN', loss_names, txt_dir=args.TBout)
                validation_logger = metrics.Logger('VALID', loss_names, txt_dir=args.TBout)
                flag_init_logger = False

            t1 = time.time()
            T1 += (t1-t0)

            if epoch <1000:
                t1 = time.time()
                config.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                config.optimizer.step()     # update the parameters
                t2 = time.time()
                T2 += (t2-t1)
            train_logger.update_epoch_logger(values_dict)
            # train_logger.print_latest_logger()
            # print('Training time distribution (seconds): loading model={:.2f}, forward={:.2f}, and backward={:.2f}'.format(T0, T1, T2))

        if config.config['epoch_decay_steps']:
            print('Cuurent lr: ', get_lr(config.optimizer))
            config.scheduler.step()

        train_logger.update_epoch_summary(epoch)#epoch)
        write_values(writer, 'train', value_dict=train_logger.get_latest_dict(), n_iter=global_step)
        write_images(writer, 'train', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

        print(separator)
        train_logger.print_latest(write_txt=True)
        print(separator)

        ''' ---------------------- 3.2 Saving model ---------------------- '''
        checkpoint = {}
        checkpoint['state_dict'] = config.model.state_dict()
        checkpoint['optimizer'] = config.optimizer.state_dict()
        checkpoint['epoch'] = epoch

        if epoch % 50 == 0:
            torch.save(checkpoint, args.model + '/model_' + str(epoch) + '.pt')
        torch.save(checkpoint, args.model + '/model_latest.pt' )

        ''' ---------------------- 3.3 Validation block ---------------------- '''
        if args.val is not None and (epoch == 1 or epoch % config.config['val_interval'] == 0):

            config.model.eval()
            dice_all_itn = 0

            for batch_idx, batch_samples in enumerate(tqdm(dataloader_val, desc='Epoch {}'.format(epoch))):
                t = time.time()
                if batch_idx < len(dataset_val):         # can specify how many samples you want to validation when training.
                    if not args.visual_euler:
                        with torch.no_grad():
                            loss_itn, images_dict, values_dict, = process_batch_opti_process(args, config, batch_samples)
                    else:
                        loss_itn, images_dict, values_dict, = process_batch_opti_process(args, config, batch_samples)

                    validation_logger.update_epoch_logger(values_dict)
                    dice_all_itn += values_dict['04_dice']

                    if False:  # if saving intermediate images during validation
                        output_dir = '{}/validation_epoch{}'.format(args.TBout, str(epoch))
                        sample_name = batch_samples['fname'][0]
                        save_images(args, images_dict, output_dir, sample_name,)
                else:
                    break

                dice= dice_all_itn / len(dataset_val)
                # print('dice_itn: ', dice)

                if dice > dice_record_0:
                    torch.save(checkpoint, args.model + '/model_best.pt')
                    dice_record_0 = dice

                config.optimizer.zero_grad(set_to_none=True)

            validation_logger.update_epoch_summary(epoch)
            write_values(writer, phase='val', value_dict=validation_logger.get_latest_dict(), n_iter=global_step)
            write_images(writer, phase='val', image_dict=images_dict, n_iter=global_step, mode3d=args.mode3d)

            print(separator)
            validation_logger.print_latest(write_txt=True)
            print(separator)


    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')


def test_single_stage(args):
    ''' ---------------------- 0. Set up  ---------------------- '''
    config = set_up_model(args)
    global_step = 0
    flag_init_logger = True
    if args.TBout is not None:
        writer = SummaryWriter('{}/tensorboard_test'.format(args.TBout))

    if args.phase == 'test_pre':
        loading_model_path = os.path.join(args.trained_model_pre, 'model_' + args.trained_model_pre_type + '.pt') if args.trained_model_pre is not None else None
    elif args.phase in ['test_post', 'test_post_multipile']:
        loading_model_path = os.path.join(args.trained_model_post, 'model_' + args.trained_model_post_type + '.pt') if args.trained_model_post is not None else None
    else:
        raise 'Wrong phase type'

    ''' ---------------------- 1. Loading pre-trained models ---------------------- '''
    previous_models = ['./model/post21/', './model/post22/', './model/CE_79/', './model/EC_1/pre/']
    _, __ = load_model(loading_model_path, config.model)#, config.optimizer, config.device)
    #

    ''' ---------------------- 2. Initializing datasets ---------------------- '''
    print(separator)
    print('Loading testing data...')
    print(separator)
    if args.dataset == 'cremi':
        dataset_test = ImageSegmentation_cremi(args, args.test, args.test_seg,
                                               softmax_data=args.test_softmax,
                                               weight_grad=args.test_ec,
                                                csv_file_betti=args.test_betti,
                                                patch_size_list=args.patch_size_list,
                                                stride_size=8,
                                                crop=False,
                                                flip=False,
                                                weight=config.config['dist'],
                                                multi=1)


    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.config['batch_size'], shuffle=False, num_workers=8, prefetch_factor=8)
    print(separator)

    ''' ---------------------- 3. Start testing ---------------------- '''
    for epoch in range(1):        #config.config['epochs']
        config.model.eval()
        # betti_list = []
        ''' ---------------------- 3.1 Testingt block ---------------------- '''
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_test, desc='Epoch {}'.format(epoch))):

            if True:#batch_idx<=10:#True: #batch_idx in [1, 2]:
                if args.visual_euler:
                    loss, images_dict, values_dict = process_batch_opti_process(args, config, batch_samples)
                else:
                    with torch.no_grad():
                        loss, images_dict, values_dict = process_batch_opti_process(args, config, batch_samples)
                # print(batch_samples['fname'], batch_samples['betti_number'])
                if flag_init_logger:
                    """only run during the first batch in the first epoch"""
                    loss_names = values_dict.keys()
                    test_logger = metrics.Logger('Test', loss_names, txt_dir=args.out)
                    flag_init_logger = False

                test_logger.update_epoch_logger(values_dict)
                if args.TBout is not None:
                    write_values(writer, 'test', value_dict=test_logger.get_latest_dict_test(), n_iter=batch_idx)
                    write_images(writer, 'test', image_dict=images_dict, n_iter=batch_idx, mode3d=args.mode3d)

                if args.out is not None :#and batch_idx <50:#     # save predictions
                    output_dir = args.out
                    sample_name = batch_samples['fname'][0]#[:-4] + '_patchsize_' + str(args.subpatch_size) + '_subpatchstride_' + str(args.subpatch_stride) + '_kernelsize_' + str(args.kernel_size) + '_sigma_' + str(args.sigma) + '_thre_' + str(args.thre) + '_binary.pkl'
                    # all images in the images_dict will be saved in the output_dir, except image and labelamp
                    save_images(args, images_dict, output_dir, sample_name, pkl=True, hide_list=['inpaianted', 'image', 'labelmap'])
            else:
                break

        test_results_all = test_logger.get_epoch_logger()
        test_logger.update_epoch_summary_test(epoch, reset=False)#epoch)
        test_results_averaged = test_logger.get_epoch_summary()

        test_tag = 'Test on pretrained model {}'.format(loading_model_path)
        with open(os.path.join(args.out, 'result_all.pkl'), 'wb') as f:
            pickle.dump({
                'info': test_tag,
                'result_all': test_results_all,
                'result_summary': test_results_averaged,
            }, f)

        print(separator)
        test_logger.print_latest_test(write_txt=True)
        print(separator)


    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')



def euler(args):
    ''' ---------------------- 0. Set up  ---------------------- '''
    config = set_up_model(args)
    global_step = 0
    flag_init_logger = True
    writer = SummaryWriter('{}/tensorboard_euler'.format(args.TBout))



    ''' ---------------------- 2. Initializing datasets ---------------------- '''
    print(separator)
    print('Loading testing data...')
    print(separator)

    dataset_test = ImageSegmentation_euler(args, args.ec_pred, args.ec_gt,
                                            csv_file_betti=args.test_betti,
                                            patch_size_list=args.patch_size_list,
                                            stride_size=8,
                                           )

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.config['batch_size'], shuffle=False, num_workers=8, prefetch_factor=8)
    print(separator)

    ''' ---------------------- 3. Start testing ---------------------- '''
    for epoch in range(1):        #config.config['epochs']
        config.ec.eval()

        ''' ---------------------- 3.1 Testing block ---------------------- '''
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_test, desc='Epoch {}'.format(epoch))):
            images_dict, values_dict = process_batch_euler(args, config, batch_samples)

            # if flag_init_logger:
            #     """only run during the first batch in the first epoch"""
            #     loss_names = values_dict.keys()
            #     test_logger = metrics.Logger('Test', loss_names, txt_dir=args.out)
            #     flag_init_logger = False
            #
            # test_logger.update_epoch_logger(values_dict)
            #
            # write_values(writer, 'test', value_dict=test_logger.get_latest_dict_test(), n_iter=batch_idx)
            # write_images(writer, 'test', image_dict=images_dict, n_iter=batch_idx, mode3d=args.mode3d)

            if args.out is not None:    # save predictions
                output_dir = args.out
                sample_name = batch_samples['fname'][0]
                # all images in the images_dict will be saved in the output_dir, except image and labelamp
                save_images(args, images_dict, output_dir, sample_name, pkl=True,  hide_list=['labelmap','image','inpaianted', 'softmax', 'argmax'])

        # test_results_all = test_logger.get_epoch_logger()
        # test_logger.update_epoch_summary_test(epoch, reset=False)#epoch)
        # test_results_averaged = test_logger.get_epoch_summary()


        # with open(os.path.join(args.TBout, 'result_all.pkl'), 'wb') as f:
        #     pickle.dump({
        #         'info': test_tag,
        #         'result_all': test_results_all,
        #         'result_summary': test_results_averaged,
        #     }, f)

        # print(separator)
        # test_logger.print_latest_test(write_txt=True)
        # print(separator)


    print(separator)
    test_tag = 'Test on prediction {} and GT {}'.format(args.ec_pred, args.ec_gt)
    print(test_tag)
    print('Finished visualization, results saved in {}'.format(output_dir))



def test_two_stages(args):
    '''End to end test the segmentation models for both stages'''
    ''' ---------------------- 0. Set up  ---------------------- '''
    config = set_up_model(args)
    global_step = 0
    flag_init_logger = True
    if args.TBout is not None:
        writer = SummaryWriter('{}/tensorboard_test'.format(args.TBout))

    assert args.phase == 'test', 'Wrong phase type'

    ''' ---------------------- 1. Loading pre-trained models ---------------------- '''

    loading_model_path_pre = os.path.join(args.trained_model_pre, 'model_' + args.trained_model_pre_type + '.pt') if args.trained_model_pre is not None else None
    loading_model_path_post = os.path.join(args.trained_model_post, 'model_' + args.trained_model_post_type + '.pt') if args.trained_model_post is not None else None


    _, __ = load_model(loading_model_path_pre, config.preNet)
    _, __ = load_model(loading_model_path_post, config.postNet)

    ''' ---------------------- 2. Initializing datasets ---------------------- '''
    print(separator)
    print('Loading testing data...')
    print(separator)

    dataset_test = ImageSegmentation_cremi(args, args.test, args.test_seg,
                                            csv_file_betti=args.test_betti,
                                            patch_size_list=args.patch_size_list,
                                            stride_size=8,
                                            crop=False,
                                            flip=False,
                                            weight=config.config['dist'],
                                            multi=1)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.config['batch_size'], shuffle=False, num_workers=8, prefetch_factor=8)
    print(separator)

    ''' ---------------------- 3. Start testing ---------------------- '''
    for epoch in range(1):        #config.config['epochs']
        config.preNet.eval()
        config.postNet.eval()

        ''' ---------------------- 3.1 Testing block ---------------------- '''
        for batch_idx, batch_samples in enumerate(tqdm(dataloader_test, desc='Epoch {}'.format(epoch))):

            args.phase = 'test_pre'
            config.update({'model': config.preNet})
            loss_pre, images_dict, values_dict = process_batch_opti_process(args, config, batch_samples)
            batch_samples.update({
            'weight_grad':  images_dict['ec'],
            'softmax_coarse':  images_dict['softmax'],
            })
            args.phase = 'test_post'
            config.model = config.postNet
            loss_post, images_dict_post, values_dict_post = process_batch_opti_process(args, config, batch_samples)

            images_dict = {'pre_' + k: v for k, v in images_dict.items()}
            values_dict = {'pre_' + k: v for k, v in values_dict.items()}

            images_dict_post = {'post_' + k: v for k, v in images_dict_post.items()}
            values_dict_post = {'post_' + k: v for k, v in values_dict_post.items()}

            images_dict.update(images_dict_post)
            values_dict.update(values_dict_post)

            if flag_init_logger:
                """only run during the first batch in the first epoch"""
                loss_names = values_dict.keys()
                test_logger = metrics.Logger('Test', loss_names, txt_dir=args.out)
                flag_init_logger = False

            test_logger.update_epoch_logger(values_dict)
            if args.TBout is not None:
                write_values(writer, 'test', value_dict=test_logger.get_latest_dict_test(), n_iter=batch_idx)
                write_images(writer, 'test', image_dict=images_dict, n_iter=batch_idx, mode3d=args.mode3d)

            if args.out is not None:    # save predictions
                output_dir = args.out
                sample_name = batch_samples['fname'][0]
                # all images in the images_dict will be saved in the output_dir, except image and labelamp
                print(images_dict.keys())
                save_images(args, images_dict, output_dir, sample_name,
                            hide_list=['pre_image', 'post_image', 'pre_labelmap', 'post_labelmap', 'pre_softmax', 'post_softmax'],
                            pkl=True)

        test_results_all = test_logger.get_epoch_logger()
        test_logger.update_epoch_summary_test(epoch, reset=False)#epoch)
        test_results_averaged = test_logger.get_epoch_summary()

        test_tag = 'Test on pretrained model {} and {}'.format(loading_model_path_pre, loading_model_path_post)
        with open(os.path.join(args.out, 'result_all.pkl'), 'wb') as f:
            pickle.dump({
                'info': test_tag,
                'result_all': test_results_all,
                'result_summary': test_results_averaged,
            }, f)

        print(separator)
        test_logger.print_latest_test(write_txt=True)
        print(separator)


    print(separator)
    print('Finished TRAINING... Plotting Graphs\n\n')

