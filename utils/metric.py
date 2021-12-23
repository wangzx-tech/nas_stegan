import numpy as np

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import lpips


def psnr(img1, img2):
    n, c, h, w = img1.shape

    img1_numpy = img1.clone().cpu().detach().numpy()
    img2_numpy = img2.clone().cpu().detach().numpy()
    img1_numpy = img1_numpy.transpose(0, 2, 3, 1)
    img2_numpy = img2_numpy.transpose(0, 2, 3, 1)

    psnr_score = np.zeros((n, 3))
    for i in range(n):
        # psnr_score[i, 0] = PSNR(img1_numpy[i, :, :, 0], img2_numpy[i, :, :, 0])
        # psnr_score[i, 1] = PSNR(img1_numpy[i, :, :, 1], img2_numpy[i, :, :, 1])
        # psnr_score[i, 2] = PSNR(img1_numpy[i, :, :, 2], img2_numpy[i, :, :, 2])
        psnr_score[i] = PSNR(img1_numpy[i], img2_numpy[i])

    return psnr_score.mean().item()


def ssim(img1, img2):
    n, c, h, w = img1.shape

    img1_numpy = img1.clone().cpu().detach().numpy()
    img2_numpy = img2.clone().cpu().detach().numpy()
    img1_numpy = img1_numpy.transpose(0, 2, 3, 1)
    img2_numpy = img2_numpy.transpose(0, 2, 3, 1)

    ssim_score = np.zeros(n)
    for i in range(n):
        ssim_score[i] = SSIM(img1_numpy[i], img2_numpy[i], multichannel=True)

    return ssim_score.mean().item()


class util_of_lpips():
    def __init__(self, net='alex', use_gpu=False):
        # Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1, img2):
        if self.use_gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()
        score_lpips = self.loss_fn.forward(img1, img2)
        return score_lpips.mean().item()
