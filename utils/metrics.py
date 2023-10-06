import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn as nn
import os

def multi_class_score(one_class_fn, predictions, labels, num_classes, one_hot=False):
    result = np.zeros(num_classes)

    for label_index in range(num_classes):
        if one_hot:
            class_predictions = predictions[:, label_index, ...]
            class_labels = labels[:, label_index, ...]
        else:
            class_predictions = predictions.eq(label_index)         # prediction [batch, 1, x,y,x]  dim2 = 0-num_class   --> class_prediction= either0 or 1
            class_predictions = class_predictions.squeeze(1)  # remove channel dim
            class_labels = labels.eq(label_index)
            class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()
        class_labels = class_labels.float()

        result[label_index] = one_class_fn(class_predictions, class_labels).mean()

    return result


def hausdorff_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    def one_class_hausdorff_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_hausdorff_distance, predictions, labels, num_classes=num_classes)


def average_surface_distance(predictions, labels, num_classes, spacing=[1, 1, 1]):
    def one_class_average_surface_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetAverageHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_average_surface_distance, predictions, labels, num_classes=num_classes)


def dice_score(predictions, labels, num_classes, one_hot=False):
    """ returns the dice score

    Args:
        predictions: one hot tensor [B, num_classes, D, H, W]
        labels: label tensor [B, 1, D, H, W]
    Returns:
        dict: ['label'] = [B, score]
    """

    def one_class_dice(pred, lab,smooth=1e-5):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()

        return (2. * true_positive+smooth) / (p_flat.sum() + l_flat.sum()+smooth)

    return multi_class_score(one_class_dice, predictions, labels, num_classes=num_classes, one_hot=one_hot)


def dice_loss(pred, lab,smooth=1e-5):
    activation = nn.Sigmoid()
    pred = activation(pred)
    shape = pred.shape
    p_flat = pred.view(shape[0], -1)
    l_flat = lab.view(shape[0], -1)
    true_positive = (p_flat * l_flat).sum()

    return 1- (2. * true_positive+ smooth) / (p_flat.sum() + l_flat.sum()+ smooth)

def precision(predictions, labels, num_classes):
    def one_class_precision(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        return true_positive / p_flat.sum()

    return multi_class_score(one_class_precision, predictions, labels, num_classes=num_classes)

def specificity(predictions, labels, num_classes):
    def one_class_specificity(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_neagtive = ((1 - p_flat) * (1- l_flat)).sum()
        false_positive = (p_flat * (1- l_flat)).sum() # labeled as negative but predict as positive
        return true_neagtive / (true_neagtive + false_positive)

    return multi_class_score(one_class_specificity, predictions, labels, num_classes=num_classes)


def recall(predictions, labels, num_classes):
    def one_class_recall(pred, lab):
        shape = pred.shape
        p_flat = pred.view(shape[0], -1)
        l_flat = lab.view(shape[0], -1)
        true_positive = (p_flat * l_flat).sum()
        negative = 1 - p_flat
        false_negative = (negative * l_flat).sum()
        return true_positive / (true_positive + false_negative)

    return multi_class_score(one_class_recall, predictions, labels, num_classes=num_classes)


class Logger():
    def __init__(self, name, loss_names, txt_dir=None):
        self.name = name
        self.loss_names = loss_names
        self.epoch_logger = {}
        self.epoch_summary = {}
        self.epoch_number_logger = []
        self.reset_epoch_logger()
        self.reset_epoch_summary()
        self.txt_dir = os.path.join(txt_dir, self.name + '.txt')

    def reset_epoch_logger(self):
        for loss_name in self.loss_names:
            self.epoch_logger[loss_name] = []

    def reset_epoch_summary(self):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name] = []

    def update_epoch_logger(self, loss_dict):
        for loss_name, loss_value in loss_dict.items():
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                self.epoch_logger[loss_name].append(loss_value)

    def update_epoch_summary(self, epoch, reset=True):
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name].append(np.mean(self.epoch_logger[loss_name]))
        self.epoch_number_logger.append(epoch)
        if reset:
            self.reset_epoch_logger()

    def update_epoch_summary_test(self, epoch, reset=True):
        # just test for one epoch, no need to make the epoch summary to be a list
        for loss_name in self.loss_names:
            self.epoch_summary[loss_name] = np.mean(self.epoch_logger[loss_name], axis=0)
        self.epoch_number_logger.append(epoch)
        if reset:
            self.reset_epoch_logger()

    def get_latest_dict(self):
        latest = {}
        for loss_name in self.loss_names:
            latest[loss_name] = self.epoch_summary[loss_name][-1]
        return latest

    def get_latest_dict_test(self):
        latest = {}
        for loss_name in self.loss_names:
            latest[loss_name] = self.epoch_logger[loss_name][-1]
        return latest

    def get_epoch_logger(self):
        return self.epoch_logger

    def get_epoch_summary(self):
        return self.epoch_summary

    def write_epoch_logger(self, location, index, loss_names, loss_labels, colours, linestyles=None, scales=None,
                           clear_plot=True):
        if linestyles is None:
            linestyles = ['-'] * len(colours)
        if scales is None:
            scales = [1] * len(colours)
        if not (len(loss_names) == len(loss_labels) and len(loss_labels) == len(colours) and len(colours) == len(
                linestyles) and len(linestyles) == len(scales)):
            raise ValueError('Length of all arg lists must be equal but got {} {} {} {} {}'.format(len(loss_names),
                                                                                                   len(loss_labels),
                                                                                                   len(colours),
                                                                                                   len(linestyles),
                                                                                                   len(scales)))

        for name, label, colour, linestyle, scale in zip(loss_names, loss_labels, colours, linestyles, scales):
            if scale == 1:
                plt.plot(range(0, len(self.epoch_logger[name])), self.epoch_logger[name], c=colour,
                         label=label, linestyle=linestyle)
            else:
                plt.plot(range(0, len(self.epoch_logger[name])), [scale * val for val in self.epoch_logger[name]],
                         c=colour,
                         label='{} x {}'.format(scale, label), linestyle=linestyle)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('{}/{}.png'.format(location, index))
        if clear_plot:
            plt.clf()

    def print_latest_logger(self, loss_names=None):
        # print_str = #'{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for i, loss_name in enumerate(loss_names):
            # print(loss_name)
            if i ==0:
                print_str = 'train_no: {}\t'.format(len(self.epoch_logger[loss_name]))
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.4f}\t'.format(loss_name, np.mean(np.array(self.epoch_logger[loss_name])))
        # print()
        # print(print_str)

    def print_latest(self, loss_names=None, write_txt=False):
        print_str = '{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for loss_name in loss_names:
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.4f}\t'.format(loss_name, self.epoch_summary[loss_name][-1])
        print(print_str)
        if write_txt:
            f = open(self.txt_dir, 'a')
            f.write(print_str)
            f.close()
            # np.savetxt(self.txt_dir )

    def print_latest_test(self, loss_names=None, write_txt=False):
        print_str = '{}\tEpoch: {}\t'.format(self.name, self.epoch_number_logger[-1])
        if loss_names is None:
            loss_names = self.loss_names
        for loss_name in loss_names:
            if loss_name not in self.loss_names:
                raise ValueError('Logger was not constructed to log {}'.format(loss_name))
            else:
                print_str += '{}: {:.4f}\t'.format(loss_name, self.epoch_summary[loss_name])
        print(print_str)

        if write_txt:
            f = open(self.txt_dir, 'a')
            f.write(print_str)
            f.close()
            # np.savetxt(self.txt_dir )