import torch
from torch.autograd import Variable
from torch.distributions import kl
import os
import numpy as np
import nibabel as nib
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import io
from barbar import Bar
from sklearn.model_selection import KFold
import random

# ----------------------------------------------------
# Some useful function to train the deep motion model
# ----------------------------------------------------

def var_or_cuda(x, device=0):
    if torch.cuda.is_available():
        x = x.cuda(device)
    return Variable(x) 

def repeat_cube(tensor, dims):
    s = [1] * len(tensor.size())
    s += dims
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = tensor.repeat(s)
    return tensor

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    # if opt.lr_policy == 'linear':
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.005, patience=3,
                                                   verbose=True, min_lr=1e-10) # threshold chosen experimentally
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def custom_load(model, path, device = 'cuda:0'):
    whole_dict = torch.load(path, map_location=device)
    try:
        model.load_state_dict(whole_dict['model'])
    except:
        # If only loading part of the model (e.g Voxelmorph)
        print("No model key in state_dict")
        del whole_dict['spatial_transform.grid']
        model.load_state_dict(whole_dict)

def custom_save(model, path):
    if type(model) == torch.nn.DataParallel:
        model = model.module

    whole_dict = {'model': model.state_dict()}
    torch.save(whole_dict, path)


def make_folds():

    # Folds for the leave-one-out

    case_list = ["case01", "case02", "case03", "case04", "case05", "case06", "case07", "case08", "case09", "case10",
                 "case11", "case12", "case13", "case14", "case15", "case16", "case17", "case18", "case19", "case20",
                 "case21", "case22", "case23", "case24", "case25"]

    kf = KFold(n_splits=25, shuffle=True, random_state=123)
    kf.get_n_splits(case_list)
    out = kf.split(case_list)
    train, valid, test = [], [], []
    random.seed(123)
    for train_idx, test_idx in out:
        train.append(np.take(case_list, train_idx))
        valid_idx = random.sample(list(train_idx), 1)
        valid.append(np.take(case_list, valid_idx))
        train[-1] = np.delete(train[-1], valid_idx)
        test.append(np.take(case_list, test_idx))
    return train, valid, test