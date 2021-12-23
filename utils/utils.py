import torch
import torch.nn.init as init
import numpy as np
import torchvision.utils as vutils


# Custom weights initialization called on netH and netR
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


def kronecker(A, B):
    C = torch.zeros(A.size(0), A.size(1), A.size(2) * B.size(0), A.size(3) * B.size(1)).cuda()
    for i in range(A.size(0)):
        for j in range(B.size(0)):
            C[i][j] = torch.einsum("ab,cd->acbd", A[i][j], B).view(A.size(2) * B.size(0), A.size(3) * B.size(1))
    return C


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic_analysis(cover, container, num_cover, secret, rev_secret, num_secret, path):
    # secret_img(b, cs*ns, h, w)    cover_img(b , cc*nc, h, w)
    batch, secret_channel, h, w = secret.size()
    channel_secret = secret_channel // num_secret
    _, cover_channel, _, _ = cover.size()
    channel_cover = cover_channel // num_cover

    cover = cover[:4]
    container = container[:4]
    secret = secret[:4]
    rev_secret = rev_secret[:4]

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap * 10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap * 10 + 0.5).clamp_(0.0, 1.0)

    for i_cover in range(num_cover):
        cover_i = cover[:, i_cover * channel_cover:(i_cover + 1) * channel_cover, :, :]
        container_i = container[:, i_cover * channel_cover:(i_cover + 1) * channel_cover, :, :]
        cover_gap_i = cover_gap[:, i_cover * channel_cover:(i_cover + 1) * channel_cover, :, :]

        if i_cover == 0:
            showCover = torch.cat((cover_i, container_i, cover_gap_i), 0)
        else:
            showCover = torch.cat((showCover, cover_i, container_i, cover_gap_i), 0)

    for i_secret in range(num_secret):
        secret_i = secret[:, i_secret * channel_secret:(i_secret + 1) * channel_secret, :, :]
        rev_secret_i = rev_secret[:, i_secret * channel_secret:(i_secret + 1) * channel_secret, :, :]
        secret_gap_i = secret_gap[:, i_secret * channel_secret:(i_secret + 1) * channel_secret, :, :]

        if i_secret == 0:
            showSecret = torch.cat((secret_i, rev_secret_i, secret_gap_i), 0)
        else:
            showSecret = torch.cat((showSecret, secret_i, rev_secret_i, secret_gap_i), 0)

    showAll = torch.cat((showCover, showSecret), 0)
    col = 3 * num_cover + 3 * num_secret
    showAll = showAll.reshape(col, 4, 3, h, w)
    showAll = showAll.permute(1, 0, 2, 3, 4)
    showAll = showAll.reshape(4 * col, 3, h, w)
    vutils.save_image(showAll, path, nrow=col, padding=1, normalize=False)
