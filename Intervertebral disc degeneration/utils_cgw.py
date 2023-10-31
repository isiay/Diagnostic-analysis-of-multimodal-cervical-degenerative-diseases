# -*- coding: utf-8 -*-
# Author: Gongwei Chen


import os
import sys
import logging
import ctypes
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict


def prepare_dir(path):
    """
    check if dir exists, if not, make dir
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def dict2order(idict, sort='key'):
    if sort == 'key':
        # building OrderedDict by sorting key
        odict = OrderedDict(sorted(idict.items(), key=lambda t: t[0]))
    elif sort == 'value':
        # building OrderedDict by sorting value
        d2 = OrderedDict(sorted(idict.items(), key=lambda t: t[1]))
    else:
        raise ValueError('No {} in Arg sort'.format(sort))


class ResultRecorder():
    """
    a class for recording and visualizing the training and validation results (Accuracy and Loss)
    """
    def __init__(self, fdir=None, fname=None, init=False):
        if init:
            self.init_func()
        else:
            self.train_loss_list = []
            self.val_loss_list = []
            self.train_acc_list = []
            self.val_acc_list = []
        self.fdir = fdir
        self.fname = fname  # file name of saved figures

    def init_func(self):
        num_epochs = 40
        self.train_loss_list = np.random.rand(num_epochs)
        self.val_loss_list = np.random.rand(num_epochs)
        self.train_acc_list = np.random.rand(num_epochs)
        self.val_acc_list = np.random.rand(num_epochs)

    def writer(self, acc=None, loss=None, phase='train'):
        if acc is not None:
            if phase == 'train':
                self.train_acc_list.append(acc)
            elif phase == 'val':
                self.val_acc_list.append(acc)
            else:
                raise NotImplementedError

        if loss is not None:
            if phase == 'train':
                self.train_loss_list.append(loss)
            elif phase == 'val':
                self.val_loss_list.append(loss)
            else:
                raise NotImplementedError

    def _visualizer(self, thist, vhist, phrase):
        assert len(thist) == len(vhist)
        if isinstance(thist, list):
            thist = np.array(thist)
            vhist = np.array(vhist)
        if thist.ndim == 2:
            thist = thist[:, 0]
            vhist = vhist[:, 0]
        if phrase == 'Acc':
            pvalue = np.max(vhist)
            px = np.argmax(vhist) + 1
        elif phrase == 'Loss':
            pvalue = np.min(vhist)
            px = np.argmin(vhist) + 1
        ymax, ymin = np.max(np.c_[thist, vhist]), np.min(np.c_[thist, vhist])
        ylimMax = max(0., 1.02 * ymax)
        ylimMin = max(0., 0.98 * ymin)
        num_epochs = len(thist)
        plt.figure()
        plt.title("Results vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel(phrase)
        plt.plot(range(1,num_epochs+1),thist,label="Train")
        plt.plot(range(1,num_epochs+1),vhist,label="Val")
        plt.plot(range(1,num_epochs+1),np.ones(num_epochs)*pvalue,label="{:.4f}".format(pvalue))
        # plt.text(0.5, pvalue*1.1, "{:.4f}".format(pvalue))
        plt.ylim((ylimMin,ylimMax))
        x_step = 10
        disp_x = np.arange(0, num_epochs+1, x_step)
        disp_x[0] = 1
        disp_x = ['{}'.format(i) if i in disp_x else '' for i in np.arange(1, num_epochs+1)]
        plt.xticks(np.arange(1, num_epochs+1), disp_x)
        plt.legend()
        if self.fdir is not None and self.fname is not None:
            plt.savefig(os.path.join(self.fdir, '{}.jpg'.format(self.fname+'_'+phrase)))
        else:
            plt.show()
        plt.close()  # need to close for freeing memory

    def visualization(self, show_content):
        if len(self.train_acc_list) == 0:
            self.init_func() 
        if show_content == 'loss':
            self._visualizer(self.train_loss_list, self.val_loss_list, 'Loss')
        elif show_content == 'acc':
            self._visualizer(self.train_acc_list, self.val_acc_list, 'Acc')
        elif show_content == 'all':
            self._visualizer(self.train_loss_list, self.val_loss_list, 'Loss')
            self._visualizer(self.train_acc_list, self.val_acc_list, 'Acc')
        if self.fdir is not None and self.fname is not None:
            np.save(os.path.join(self.fdir, '{}.npy'.format(self.fname)), {'train_acc': self.train_acc_list,
            'train_loss': self.train_loss_list, 'val_acc': self.val_acc_list, 'val_loss': self.val_loss_list})

 
class LogFormatter(logging.Formatter):
    """
    color format for different log level
    Coping from https://github.com/ycszen/TorchSeg/blob/master/furnace/engine/logger.py
    """
    def __init__(self, log_fout):
        super().__init__()
        self.log_fout = log_fout   # whether output in a file
        # self.date_full = '[%(asctime)s %(lineno)d@%(filename)s] '
        self.date_full = '[%(asctime)s] '
        self.date = '%(asctime)s '
        self.msg = '%(message)s'
        self.datefmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, '[DBG]'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, '[WRN]'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, '[ERR]'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
        else:
            self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_warn(msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_err(msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_omitted(msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)


class Logger():
    """
    a class for print infromation on console or recording information in a log file
    Level: critical > error > warning > info > debug,notset
    """
    def __init__(self, lname=None, fpath=None, clevel=logging.INFO, flevel=logging.DEBUG):
        if lname is not None:
            self.logger = logging.getLogger(lname)
        else:
            # when name is not specific, root logger will be returned. 
            self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set console log
        ch = logging.StreamHandler()
        ch.setFormatter(LogFormatter(False))
        ch.setLevel(clevel)
        self.logger.addHandler(ch)
        # set file log
        if fpath is not None:
            fh = logging.FileHandler(fpath)
            fh.setFormatter(LogFormatter(True))
            fh.setLevel(flevel)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.allnum = 0
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes))

    def update(self, output, target):
        # output, target need to be numpy array
        self.allnum += len(target)
        pred = output.argmax(axis=1)
        for i, j in zip(target, pred):
            self.cm[i, j] += 1


def topk_acc(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pytorch_deterministic(rand_seed=1234):
    # fix random seed for reproducibility, also need to make cudnn.deterministic = True
    # and cudnn.benchmark = False
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


from scipy.spatial.kdtree import KDTree
import time

def locally_extreme_points(coords, data, neighbourhood, lookfor='max', p_norm=2.):
    '''
    Find local maxima of points in a pointcloud.  Ties result in both points passing through the filter.

    Not to be used for high-dimensional data.  It will be slow.

    coords: A shape (n_points, n_dims) array of point locations
    data: A shape (n_points, ) vector of point values
    neighbourhood: The (scalar) size of the neighbourhood in which to search.
    lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima
    p_norm: The p-norm to use for measuring distance (e.g. 1=Manhattan, 2=Euclidian)

    returns
        filtered_coords: The coordinates of locally extreme points
        filtered_data: The values of these points
    '''
    assert coords.shape[0] == data.shape[0], 'You must have one coordinate per data point'
    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_ball_tree(kdtree, r=neighbourhood, p = p_norm)
    i_am_extreme = [data[i]==extreme_fcn(data[n]) for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme)  # This line just saves time on indexing
    return coords[extrema], data[extrema]


if __name__ == '__main__':

    """
    Test for ResultRecorder
    """
    # myrecorder = ResultRecorder(init=True)
    # myrecorder.visualization('loss')
    
    # Test logger
    # mylog = Logger('my.log')
    # mylog.debug('a debug message')
    # mylog.info('an info message')
    # mylog.warn('a warning message')

    # Test Look extreme points
    h = 5
    np.random.seed(123)
    a = np.random.randn(h, h)
    b = np.meshgrid(np.arange(h), np.arange(h))
    b = np.stack(b, axis=-1)
    a = a.reshape([-1])
    b = b.reshape([-1, 2])
    start_t = time.time()
    for _ in range(256*32):
        c_i, c = locally_extreme_points(b, a, 1.5)  # 1.5 with euclidean dist for 8-neighbor
    print((time.time() - start_t))
    print((time.time() - start_t) / (256*32.))
