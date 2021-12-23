import os
import sys
import time
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import network.revealnet
from network.revealnet import UnetR
from network.network import Network
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, weights_init

parser = argparse.ArgumentParser("search")

# Train configuration.
parser.add_argument('--gpu', type=int, default='0', help='gpu device id')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--begin', type=int, default=15, help='begin epoch for update arch weights')
parser.add_argument('--imagesize', type=int, default='32', help='image resize')
parser.add_argument('--dataset', type=str, default='cifar10', help='datasets use to search')

# Model configuration.
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# Directories.
parser.add_argument('--root_cifar', type=str, default='datasets', help='cifar root dir')
parser.add_argument('--data', type=str, default='datasets/Data/CLS-LOC', help='location of the data corpus')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='output/checkpoint', help='experiment name')

# Miscellaneous.
parser.add_argument('--workers', type=int, default=0, help='number of workers to load dataset')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

args = parser.parse_args()

args.save = '{}search-{}-init_channel{}-layer{}'\
    .format(args.save, time.strftime("%Y%m%d-%H"), args.init_channels, args.layers)
if not os.path.exists(args.save):
    os.makedirs(args.save)

# config log
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # Improve speed
    np.random.seed(args.seed)
    cudnn.benchmark = True
    # Reproducible
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    # Tensorboard
    writer_file = args.save + '/tb'
    if not os.path.exists(writer_file):
        os.makedirs(writer_file)
    writer = SummaryWriter(log_dir=writer_file)

    # logs for config
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    # loss
    criterion = nn.MSELoss().cuda()

    # dataset prepare
    transform_image = transforms.Compose([
            transforms.Resize([args.imagesize, args.imagesize]),
            transforms.ToTensor(),
        ])
    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(root=args.root_cifar, train=True, download=True, transform=transform_image)
        valid_data = dset.CIFAR10(root=args.root_cifar, train=True, download=True, transform=transform_image)
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_data = dset.ImageFolder(traindir, transform_image)
        valid_data = dset.ImageFolder(valdir, transform_image)

    # arguments for search network
    model = Network(args.init_channels, args.layers, criterion)
    Rnet = UnetR(input_nc=3,
                 output_nc=3,
                 nhf=args.init_channels,
                 norm_layer=nn.BatchNorm2d,
                 output_function=nn.Sigmoid)
    model = torch.nn.DataParallel(model).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()
    Rnet.apply(weights_init)

    # optimizer
    params = list(model.parameters()) + list(Rnet.parameters())
    optimizer = optim.Adam(params=params, lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_arch = optim.Adam(params=model.module.arch_parameters(), lr=args.arch_learning_rate,
                                betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

    # dataloader
    train_queue_cover = DataLoader(train_data, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.workers, pin_memory=True)
    train_queue_secret = DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.workers, pin_memory=True)

    valid_queue_cover = DataLoader(valid_data, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_queue_secret = DataLoader(valid_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.workers, pin_memory=True)

    small_loss = 1000
    for epoch in range(args.epochs):

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

        arch_param = model.module.arch_parameters()
        logging.info(F.softmax(arch_param[0], dim=-1))
        logging.info(F.softmax(arch_param[1], dim=-1))

        # training
        train_queue = zip(train_queue_cover, train_queue_secret)
        valid_queue = zip(valid_queue_cover, valid_queue_secret)
        train_l2loss = train(train_queue, valid_queue, model, Rnet, optimizer, optimizer_arch, criterion, epoch)
        logging.info(f'epoch {epoch} train_l2loss {train_l2loss}')

        train_loss = train_l2loss
        scheduler.step(train_loss)

        # Save the best model parameters
        if train_loss < small_loss:
            small_loss = train_loss
            filename = f'{args.save}/checkpoint.pth.tar'
            state = {
                'genotype': genotype,
            }
            torch.save(state, filename)

        writer.add_scalar('train_l2loss', train_loss, epoch)
    writer.close()


def train(train_queue, valid_queue, model, reveal_net, optimizer, optimizer_arch, criterion, epoch):
    l2loss = AverageMeter()
    model.train()
    reveal_net.train()

    for step, ((cover_img, cover_target), (secret_img, secret_target)) in enumerate(train_queue):

        cover_img = cover_img.cuda(non_blocking=True)
        secret_img = secret_img.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        (cover_img_search, cover_img_search_target), (secret_img_search, secret_img_search_target) = \
            next(iter(valid_queue))
        cover_img_search = cover_img_search.cuda(non_blocking=True)
        secret_img_search = secret_img_search.cuda(non_blocking=True)

        # x个epoch之后每个batch更新一次架构参数
        if epoch >= args.begin:
            optimizer_arch.zero_grad()
            stega_img_search = model(secret_img_search) + cover_img_search
            loss_a_hide = criterion(stega_img_search, cover_img_search)

            reveal_img_search = reveal_net(stega_img_search)
            loss_a_reveal = criterion(reveal_img_search, secret_img_search)

            loss_a = loss_a_hide + 0.75 * loss_a_reveal
            loss_a.sum().backward()

            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_arch.step()

        optimizer.zero_grad()
        stega_img = model(secret_img) + cover_img
        loss_hide = criterion(stega_img, cover_img)

        reveal_img = reveal_net(stega_img)
        loss_reveal = criterion(secret_img, reveal_img)

        loss = loss_hide + 0.75 * loss_reveal
        loss.backward()

        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)  # 梯度截断
        optimizer.step()

        # update loss
        l2loss.update(loss.item(), args.batch_size)

        if step % args.report_freq == 0:
            logging.info(f'epoch {epoch} train {step} l2loss {l2loss.avg}')

    return l2loss.avg


if __name__ == '__main__':
    main()
