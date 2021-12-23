# encoding: utf-8

import argparse
import os
import sys
import time

import numpy as np
import torch.cuda
from torch import nn, optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from network import genotypes
from network.revealnet import UnetR
from network.hidenet import UnetG, NASG
from solver import Solver
from utils.utils import weights_init
from utils.log import print_net, save_log

parser = argparse.ArgumentParser()

# Training configuration.
parser.add_argument('--gpu', type=str, default='0', help='cuda device to use')
parser.add_argument('--batchsize', type=int, default=44, help='input batch size')
parser.add_argument('--iters_1epoch', type=int, default=2000, help='iteration each epoch')
parser.add_argument('--num_epochs', type=int, default=65, help='epochs for training')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='choose mode')
parser.add_argument('--ckpt_use_train', action='store_true', default=False, help='whether use checkpoint to continue train')

# Network configuration
parser.add_argument('--init_channels', type=int, default=64)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--num_downs', type=int, default=5)

# Data configuration
parser.add_argument('--num_secret_channel', type=int, default=3)
parser.add_argument('--num_cover_channel', type=int, default=3)
parser.add_argument('--num_secret', type=int, default=1)
parser.add_argument('--num_cover', type=int, default=1)
parser.add_argument('--imagesize', type=int, default=128)

# Model configuration.
parser.add_argument('--loss', type=str, default='l2')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
parser.add_argument('--beta', type=int, default=0.75, help='beta weights for LossR')
parser.add_argument('--beta1', type=int, default=0.5, help='beta1 for adam')
parser.add_argument('--norm', type=str, default='batch', choices=['instance', 'batch', 'None'], help='which normalization to use')
parser.add_argument('--net', type=str, default='NAS', choices=['UNet', 'NAS'], help='choose the network for train/test')
parser.add_argument('--arch', type=str, default='NAS_StegaNet', help='architecture when net is NAS')

# Test configuration.
parser.add_argument('--test_ckpt', type=str, default='')

# Miscellaneous.
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--use_tensorboard', type=bool, default=True, help='whether use tensorboard')
parser.add_argument('--cleanlog', type=bool, default=True, help='whether clear log when codes begin')

# Directories.
parser.add_argument('--train_data_dir', type=str, default='datasets/ILSVRC/Data/CLS-LOC/train')
parser.add_argument('--val_data_dir', type=str, default='datasets/ILSVRC/Data/CLS-LOC/val')
parser.add_argument('--test_data_dir', type=str, default='datasets/ILSVRC/Data/CLS-LOC/val')
parser.add_argument('--checkpoint', type=str, default='output/checkpoint')

config = parser.parse_args()
config.checkpoint = '{}train-{}-init_channel{}-layer{}' \
    .format(config.checkpoint, time.strftime("%Y%m%d-%H"), config.init_channels, config.layers)
print(config)

# Set CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu


def main():
    # For fast training
    if torch.cuda.is_available():
        cudnn.benchmark = True
        np.random.seed(2)
        torch.manual_seed(2)
    else:
        sys.exit(1)

    # Create directories if not exist
    if not os.path.exists(config.checkpoint) and config.mode == 'train':
        os.makedirs(config.checkpoint)
    log_dir = config.checkpoint + '/logs'
    log_file = config.checkpoint + '/logs/log.txt'
    if not os.path.exists(log_dir) and config.mode == 'train':
        os.makedirs(log_dir)

    # clear log.txt and save configuration when train from scratch
    if config.mode == 'train' and not config.ckpt_use_train and config.cleanlog:
        if os.path.exists(log_file):
            with open(log_file, 'a+', encoding='utf-8') as log:
                log.truncate(0)
    if config.mode == 'train':
        save_log(str(config), log_file)

    # Nomalization
    if config.norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif config.norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        norm_layer = None

    # Network Config
    if config.net == 'UNet':
        Hnet = UnetG(input_nc=config.num_secret_channel * config.num_secret,
                     output_nc=config.num_cover_channel * config.num_cover,
                     num_downs=config.num_downs,
                     ngf=config.init_channels,
                     norm_layer=norm_layer,
                     use_dropout=False,
                     out_func=nn.Tanh)
        # Net weights initialization
        Hnet.apply(weights_init)

    if config.net == 'NAS':
        genotype = eval(f'genotypes.{config.arch}')
        if config.mode == 'train':
            save_log(str(genotype), log_file)
        Hnet = NASG(config.init_channels, config.layers, genotype,
                    config.num_secret_channel * config.num_secret,
                    config.num_cover_channel * config.num_cover)

    Rnet = UnetR(input_nc=config.num_cover_channel * config.num_cover,
                 output_nc=config.num_secret_channel * config.num_secret,
                 nhf=config.init_channels,
                 norm_layer=norm_layer,
                 output_function=nn.Sigmoid)
    # Net weights initialization
    Rnet.apply(weights_init)

    if config.mode == 'train':
        print_net(Hnet, log_file, torch.randn(config.batchsize, config.num_secret_channel * config.num_secret, config.imagesize, config.imagesize))
        print_net(Rnet, log_file, torch.randn(config.batchsize, config.num_cover_channel * config.num_cover, config.imagesize, config.imagesize))
        ##### Always set to multiple GPU mode  #####
        Hnet = torch.nn.DataParallel(Hnet).cuda()
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    else:
        Hnet = Hnet.cuda()
        Rnet = Rnet.cuda()

    # loss
    if config.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    if config.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    # config optimizer
    params = list(Hnet.parameters()) + list(Rnet.parameters())
    optimizer = optim.Adam(params, lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

    # Data Loader
    transforms_color = transforms.Compose([
        transforms.Resize([config.imagesize, config.imagesize]),
        transforms.ToTensor(),
    ])

    transforms_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([config.imagesize, config.imagesize]),
        transforms.ToTensor(),
    ])

    if config.num_cover_channel == 1:
        transforms_cover = transforms_gray
    elif config.num_cover_channel == 3:
        transforms_cover = transforms_color
    if config.num_secret_channel == 1:
        transforms_secret = transforms_gray
    elif config.num_secret_channel == 3:
        transforms_secret = transforms_color

    # Solver for training and testing
    if config.mode == 'train':
        # continue train
        if config.ckpt_use_train:
            config.checkpoint = config.test_ckpt
        # Load train data and validation data for train
        train_loader_cover = DataLoader(ImageFolder(config.train_data_dir, transform=transforms_cover),
                                        batch_size=config.batchsize * config.num_cover,
                                        shuffle=True,
                                        num_workers=config.workers)
        train_loader_secret = DataLoader(ImageFolder(config.train_data_dir, transform=transforms_secret),
                                         batch_size=config.batchsize * config.num_secret,
                                         shuffle=True,
                                         num_workers=config.workers)
        val_loader_cover = DataLoader(ImageFolder(config.val_data_dir, transform=transforms_cover),
                                      batch_size=config.batchsize * config.num_cover,
                                      shuffle=True,
                                      num_workers=config.workers)
        val_loader_secret = DataLoader(ImageFolder(config.val_data_dir, transform=transforms_secret),
                                       batch_size=config.batchsize * config.num_secret,
                                       shuffle=False,
                                       num_workers=config.workers)

        solver = Solver(config=config,
                        optimizer=optimizer,
                        Hnet=Hnet,
                        Rnet=Rnet,
                        criterion=criterion,
                        scheduler=scheduler)
        solver.train_val(train_cover=train_loader_cover,
                         train_secret=train_loader_secret,
                         val_cover=val_loader_cover,
                         val_secret=val_loader_secret)

    elif config.mode == 'test':
        # Set up test environment and modal file used
        config.checkpoint = config.test_ckpt
        config.gpu = int(0)
        # Load test data for test
        test_loader_cover = DataLoader(ImageFolder(config.test_data_dir, transform=transforms_cover),
                                       batch_size=config.batchsize * config.num_cover,
                                       shuffle=True,
                                       num_workers=config.workers)
        test_loader_secret = DataLoader(ImageFolder(config.test_data_dir, transform=transforms_secret),
                                        batch_size=config.batchsize * config.num_secret,
                                        shuffle=True,
                                        num_workers=config.workers)

        solver = Solver(config=config,
                        optimizer=optimizer,
                        Hnet=Hnet,
                        Rnet=Rnet,
                        criterion=criterion,
                        scheduler=scheduler)
        solver.test(test_secret=test_loader_secret,
                    test_cover=test_loader_cover)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
