
#!/usr/bin/env python
# coding=utf-8

import os
from config import Config

opt = Config('trmash.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

import random
import time
import numpy as np
from pathlib import Path

import utils
from data_RGB import get_training_data, get_validation_data
from MHNet import MHNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx


dir_checkpoint = Path('./mhnetmash/')

def train():

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir   = opt.TRAINING.VAL_DIR

    ######### Model ###########
    model_restoration = MHNet()
    print("Total number of param  is ", sum(x.numel() for x in model_restoration.parameters()))
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
      print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)


    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest    = './mhnetmash/model_best.pth'
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)
        print("duoka")

    ######### Loss ###########
    criterion_mse = losses.PSNRLoss()
    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)



    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        psnr_train_rgb = []
        psnr_train_rgb1 = []
        psnr_tr = 0
        psnr_tr1 = 0
        model_restoration.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            restored = model_restoration(input_)

            loss = criterion_mse(restored[0],target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step = global_step+1

        psnr_te = 0
        psnr_te_1 = 0
        ssim_te_1 = 0
        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            psnr_val_rgb1 = []
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restore = restored[0]

                for res, tar in zip(restore, target):
                    tssss = utils.torchPSNR(res, tar)
                    psnr_te = psnr_te + tssss
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            print("te", psnr_te)

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, str(dir_checkpoint / "model_best.pth"))


            print("[epoch %d PSNR: %.4f best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")


if __name__=='__main__':
    train()

