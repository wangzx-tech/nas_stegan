import itertools
import os
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
import time

import torch

from utils.log import save_log
from utils.metric import psnr, ssim
from utils.metric import util_of_lpips
from utils.utils import AverageMeter, save_result_pic_analysis


def forward_pass(secret_img, cover_img, batchsize, Hnet, Rnet, criterion):
    # secret_img(b * ns, c, h, w)    cover_img(b * nc, c, h, w)
    secret_batch, secret_channel, h, w = secret_img.size()
    num_secret = secret_batch // batchsize
    cover_batch, cover_channel, _, _ = cover_img.size()
    num_cover = cover_batch // batchsize
    secret_img = secret_img.cuda()
    cover_img = cover_img.cuda()

    # secret_img(b, c*ns, h, w)    cover_img(b , c*nc, h, w)
    secret_img = secret_img.view(batchsize, secret_channel * num_secret, h, w)
    cover_img = cover_img.view(batchsize, secret_channel * num_cover, h, w)

    # Get container images
    H_input = secret_img
    itm_secret_img = Hnet(H_input)
    stega_img = itm_secret_img + cover_img

    # Get reveal images
    rev_secret_img = Rnet(stega_img)

    # Get loss
    errH = criterion(stega_img, cover_img)
    errR = criterion(rev_secret_img, secret_img)
    errH_l1 = itm_secret_img.abs().mean() * 255
    errR_l1 = (rev_secret_img - secret_img).abs().mean() * 255

    return cover_img, stega_img, secret_img, rev_secret_img, errH, errR, errH_l1, errR_l1


class Solver:
    def __init__(self, config=None, optimizer=None, Hnet=None, Rnet=None, criterion=None, scheduler=None):
        # config
        self.lr = config.lr
        self.batchsize = config.batchsize
        self.num_cover = config.num_cover
        self.num_secret = config.num_secret
        self.beta = config.beta
        self.checkpoint = config.checkpoint
        self.epochs = config.num_epochs
        self.ckpt_use_train = config.ckpt_use_train
        self.use_tensorboard = config.use_tensorboard
        self.use_pretrain = config.use_pretrain
        self.gpu = config.gpu
        self.iters_1epoch = config.iters_1epoch

        self.optimizer = optimizer
        self.Hnet = Hnet
        self.Rnet = Rnet
        self.criterion = criterion
        self.scheduler = scheduler

        self.log_path = config.checkpoint + '/logs/log.txt'

        if self.use_tensorboard:
            writer_file = config.checkpoint + '/tb'
            if not os.path.exists(writer_file):
                os.makedirs(writer_file)
            self.writer = SummaryWriter(log_dir=writer_file)

    def train_val(self, train_cover, train_secret, val_cover, val_secret):
        small_loss = 10000
        epoch_begin = 0

        # Continue train
        if self.ckpt_use_train:
            if self.checkpoint != '':
                checkpoint = torch.load(self.checkpoint + '/checkpoint.pth.tar', map_location=f'cuda:{self.gpu}')
                epoch_begin = checkpoint['epoch']
                self.Hnet.load_state_dict(checkpoint['H_state_dict'])
                self.Rnet.load_state_dict(checkpoint['R_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        for epoch in range(epoch_begin, self.epochs):
            # update learning rate
            self.adjust_learning_rate(epoch)

            train_loader = zip(train_secret, train_cover)
            val_loader = zip(val_secret, val_cover)

            # Train
            self.train(train_loader, epoch)
            # Validation
            with torch.no_grad():
                Hlosses, Rlosses, Hl1loss, Rl1loss = self.validation(val_loader, epoch)

            # adjust learning rate ###
            self.scheduler.step(Rlosses)

            # Save the best model parameters
            sum_losses = Hl1loss + Rl1loss
            if sum_losses < small_loss:
                small_loss = sum_losses
                filename = '%s/checkpoint.pth.tar' % self.checkpoint
                state = {
                    'epoch': epoch + 1,
                    'H_state_dict': self.Hnet.module.state_dict(),
                    'R_state_dict': self.Rnet.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(state, filename)

        if self.use_tensorboard:
            self.writer.close()

    def train(self, train_loader, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        Hlosses = AverageMeter()
        Rlosses = AverageMeter()
        errH_l1loss = AverageMeter()
        errR_l1loss = AverageMeter()

        print("################################# train begin #############################################")

        # switch to train mode
        self.Hnet.train()
        self.Rnet.train()

        start_time = time.time()

        for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(train_loader, 0):

            data_time.update(time.time() - start_time)

            cover_img, stega_img, secret_img, rec_secret_img, errH, errR, errH_l1, errR_l1 \
                = forward_pass(secret_img, cover_img, self.batchsize, self.Hnet, self.Rnet, self.criterion)

            Hlosses.update(errH.item(), self.batchsize * self.num_cover)
            Rlosses.update(errR.item(), self.batchsize * self.num_secret)
            errH_l1loss.update(errH_l1.item(), self.batchsize * self.num_cover)
            errR_l1loss.update(errR_l1.item(), self.batchsize * self.num_secret)

            # Loss, backpropagation, and optimization step
            sum_loss = errH + self.beta * errR
            self.optimizer.zero_grad()
            sum_loss.backward()
            self.optimizer.step()

            # Time spent on one batch
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            # print log
            log = f'epochs: {epoch}/{self.epochs} iters: {i} train_LossH: {Hlosses.val} train_LossR: {Rlosses.val} ' \
                  f'train_LossH_l1={errH_l1loss.val} train_LossR_l1={errR_l1loss.val} \n' \
                  f'datatime: {data_time.val} batchtime: {batch_time.val} \n'
            if i % 100 == 0:
                print(log)

            if i == self.iters_1epoch:
                break

        epoch_log = f"Training {epoch} train_LossH={Hlosses.avg} train_LossR={Rlosses.avg} " \
                    f"train_LossH_l1={errH_l1loss.avg} train_LossR_l1={errR_l1loss.avg} \n" \
                    f"lr= {self.optimizer.param_groups[0]['lr']} Epoch time= {batch_time.sum}"
        print(epoch_log)
        save_log(epoch_log, self.log_path)

        if self.use_tensorboard:
            self.writer.add_scalar("lr/lr", self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
            self.writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
            self.writer.add_scalar('train/H_l1loss', errH_l1loss.avg, epoch)
            self.writer.add_scalar('train/R_l1loss', errR_l1loss.avg, epoch)

        print("###################################### train end #####################################")

    def validation(self, val_loader, epoch):
        print("############################## validation begin #################################")

        start_time = time.time()
        self.Hnet.eval()
        self.Rnet.eval()
        batch_time = AverageMeter()
        Hlosses = AverageMeter()
        Rlosses = AverageMeter()
        errH_l1loss = AverageMeter()
        errR_l1loss = AverageMeter()

        for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(val_loader, 0):

            cover_img, stega_img, secret_img, rec_secret_img, errH, errR, errH_l1, errR_l1 \
                = forward_pass(secret_img, cover_img, self.batchsize, self.Hnet, self.Rnet, self.criterion)

            Hlosses.update(errH.item(), self.batchsize * self.num_cover)
            Rlosses.update(errR.item(), self.batchsize * self.num_secret)
            errH_l1loss.update(errH_l1.item(), self.batchsize * self.num_cover)
            errR_l1loss.update(errR_l1.item(), self.batchsize * self.num_secret)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            val_log = f'epochs: {epoch}/{self.epochs} iters: {i} val_LossH: {Hlosses.val} val_LossR: {Rlosses.val} \n' \
                      f'val_LossH_l1 = {errH_l1loss.val} val_LossR_l1={errR_l1loss.val} batchtime: {batch_time.val} \n'
            if i % 30 == 0:
                print(val_log)
            if i == 150:
                break

        val_log = f"validation {epoch} val_LossH = {Hlosses.avg} val_LossR = {Rlosses.avg}  \n" \
                  f"val_LossH_l1 = {errH_l1loss.avg} val_LossR_l1={errR_l1loss.avg} validation time={batch_time.sum}\n"
        print(val_log)
        save_log(val_log, self.log_path)

        if self.use_tensorboard:
            self.writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
            self.writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
            self.writer.add_scalar('validation/H_diff_avg', errH_l1loss.avg, epoch)
            self.writer.add_scalar('validation/R_diff_avg', errR_l1loss.avg, epoch)

        print("################################ validation end ###################################\n")

        return Hlosses.avg, Rlosses.avg, errH_l1loss.avg, errR_l1loss.avg

    def test(self, test_secret, test_cover):
        test_loader = itertools.zip(test_secret, test_cover)

        batch_time = AverageMeter()

        psnr_c = AverageMeter()
        psnr_s = AverageMeter()
        ssim_c = AverageMeter()
        ssim_s = AverageMeter()
        lpips_c = AverageMeter()
        lpips_s = AverageMeter()
        APD_c = AverageMeter()
        APD_s = AverageMeter()

        print("#################################### test begin #########################################")

        if self.checkpoint != "":

            checkpoint = torch.load(self.checkpoint + '/checkpoint.pth.tar', map_location=f'cuda:{self.gpu}')
            self.Hnet.load_state_dict(checkpoint['H_state_dict'], strict=False)
            self.Rnet.load_state_dict(checkpoint['R_state_dict'], strict=False)

        self.Hnet.eval()
        self.Rnet.eval()

        for i, ((secret_img, secret_target), (cover_img, cover_target)) in enumerate(test_loader, 0):

            start_time = time.time()
            # Test
            with torch.no_grad():
                cover_img, stega_img, secret_img, rec_secret_img, errH, errR, errH_l1, errR_l1 \
                    = forward_pass(secret_img, cover_img, self.batchsize, self.Hnet, self.Rnet, self.criterion)

            batch_time.update(time.time() - start_time)

            # Metrics for test result
            # Config lpips
            lpips = util_of_lpips(net='alex', use_gpu=True)
            # quantitative results for cover images and container images
            psnr_c.update(psnr(cover_img, stega_img), self.batchsize * self.num_cover)
            ssim_c.update(ssim(cover_img, stega_img), self.batchsize * self.num_cover)
            if self.num_cover == 1:
                lpips_c.update(lpips.calc_lpips(cover_img, stega_img), self.batchsize * self.num_cover)
            else:
                for num in range(self.num_secret):
                    lpips_s.update(lpips.calc_lpips(secret_img[:, num * 3:(num + 1) * 3, :, :],
                                                    rec_secret_img[:, num * 3:(num + 1) * 3, :, :]), self.batchsize)
            APD_c.update(errH_l1.item(), self.batchsize * self.num_cover)
            # quantitative results for secret images and reveal images
            psnr_s.update(psnr(secret_img, rec_secret_img), self.batchsize * self.num_secret)
            ssim_s.update(ssim(secret_img, rec_secret_img), self.batchsize * self.num_secret)
            if self.num_secret == 1:
                lpips_s.update(lpips.calc_lpips(secret_img, rec_secret_img), self.batchsize * self.num_secret)
            else:
                for num in range(self.num_secret):
                    lpips_s.update(lpips.calc_lpips(secret_img[:, num * 3:(num + 1) * 3, :, :],
                                                    rec_secret_img[:, num * 3:(num + 1) * 3, :, :]), self.batchsize)
            APD_s.update(errR_l1.item(), self.batchsize * self.num_secret)

            # save test results
            outimg_dir = os.path.join(self.checkpoint, 'img')
            if not os.path.exists(outimg_dir):
                os.makedirs(outimg_dir)
            filename = outimg_dir + '/test' + str(i) + '.jpg'
            save_result_pic_analysis(cover_img, stega_img, self.num_cover,
                                     secret_img, rec_secret_img, self.num_secret, filename)

            print(
                f'PSNR C: {psnr_c.val} , SSIM C: {ssim_c.val} , LPIPS C: {lpips_c.val} , APD C: {APD_c.val} ,\n '
                f'PSNR S: {psnr_s.val} , SSIM S: {ssim_s.val} , LPIPS S: {lpips_s.val} , APD S: {APD_s.val} \n'
                f'Batch_time :{batch_time.val} \n'
            )
            if i == 10:
                break
        print(
            f'PSNR C: {psnr_c.avg} , SSIM C: {ssim_c.avg} , LPIPS C: {lpips_c.avg} , APD C: {APD_c.avg} ,\n '
            f'PSNR S: {psnr_s.avg} , SSIM S: {ssim_s.avg} , LPIPS S: {lpips_s.avg} , APD S: {APD_s.avg} \n'
            f'Batch_time :{batch_time.avg} \n'
        )
        print("################################## test end ########################################")

    def adjust_learning_rate(self, epoch):
        # update learning rate each 30 epochs
        lr = self.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            print(f'learning rate:{lr}')
