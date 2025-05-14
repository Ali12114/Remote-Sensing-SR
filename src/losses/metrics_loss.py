import piq

from metrics import _cc_single_torch
from utils import load_fun
import torch
import torch.nn as nn
import numpy as np
import losses.swt_losses as wt_loss
from pytorch_msssim import ms_ssim


def norm_0_to_1(fun):
    def wrapper(cfg):
        dset = cfg.dataset
        use_minmax = cfg.dataset.get('stats').get('use_minmax', False)
        denorm = load_fun(dset.get('denorm'))(
                cfg,
                hr_name=cfg.dataset.hr_name,
                lr_name=cfg.dataset.lr_name)
        evaluable = load_fun(dset.get('printable'))(
                cfg,
                hr_name=cfg.dataset.hr_name,
                lr_name=cfg.dataset.lr_name,
                filter_outliers=False,
                use_minmax=use_minmax)

        rfun = fun(cfg)

        def f(sr, hr):
            hr, sr, _ = evaluable(*denorm(hr, sr, None))
            sr = sr.clamp(min=0, max=1)
            hr = hr.clamp(min=0, max=1)
            return rfun(sr, hr)

        return f
    return wrapper


@norm_0_to_1
def ssim_loss(cfg):
    criterion = piq.SSIMLoss()
    # criterion = PSNRLoss()
    # criterion = VGGPerceptualLoss().to('cuda')

    def f(sr, hr):
        return criterion(sr, hr)

    return f

@norm_0_to_1
def perceptual_loss(cfg):
    # criterion = piq.SSIMLoss()
    # criterion = PSNRLoss()
    criterion = VGGPerceptualLoss().to('cuda')

    def f(sr, hr):
        return criterion(sr, hr)

    return f

@norm_0_to_1
def swt_loss(cfg):
    criterion = wt_loss.SWTLoss().to('cuda')

    def f(sr, hr):
        return criterion(sr, hr)

    return f


def cc_loss(sr, hr):
    # print(sr.shape)
    # print(hr.shape)
    cc_value = _cc_single_torch(sr, hr)
    return 1 - ((cc_value + 1) * .5)
