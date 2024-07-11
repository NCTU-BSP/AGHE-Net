# -*- coding: utf-8 -*-
import json
import warnings
from argparse import ArgumentParser

import torch
torch.cuda.set_device('cuda:0')

from modules import train3d

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()
    # for model
    parser.add_argument('--model', type=str, required=False, default='')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--init_ch', type=int, default=32)
    parser.add_argument('--deconv', action='store_true', default=False)
    parser.add_argument('--weight_init', type=str, default='kaiming')
    parser.add_argument('--cr_cpt_path', type=str)

    # for UNet3P
    parser.add_argument('--reduce_ch', type=int, default=64)

    # for RUNet, R2UNet
    parser.add_argument('--num_rcnn', type=int, default=2)
    parser.add_argument('--t', type=int, default=2)

    # for classification guided module (multi-task)
    parser.add_argument('--cgm', action='store_true', default=True)
    parser.add_argument('--cgm_weight', type=float, default=0.2)

    # for focal tversky loss
    parser.add_argument('--ft_alpha', type=float, default=0.7)
    parser.add_argument('--ft_beta', type=float, default=0.3)
    parser.add_argument('--ft_gamma', type=float, default=0.75)

    # for unified focal loss
    parser.add_argument('--uf_weight', type=float, default=0.5)
    parser.add_argument('--uf_delta', type=float, default=0.6)
    parser.add_argument('--uf_gamma', type=float, default=0.2)
    parser.add_argument('--uf_loss_weight', type=float, default=1.0)

    # for boundary loss
    parser.add_argument('--bd_loss', action='store_true', default=True)
    parser.add_argument('--bd_loss_weight', type=float, default=0.2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=300)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--criterion', type=str, required=False, default='UnifiedFocalLoss')
    parser.add_argument('--lr', type=float, default=2e-4)

    # for lr scheduler
    parser.add_argument('--reduce_lr_on_plateau', action='store_true', default=False)
    parser.add_argument('--exponential_lr', action='store_true', default=False)
    parser.add_argument('--step_lr', action='store_true', default=False)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='ncct111_gt-1_5fold')
    parser.add_argument('--cv', type=int, required=False, default=1)
    parser.add_argument('--patch_x', type=int, required=False, default=128)
    parser.add_argument('--patch_y', type=int, required=False, default=128)
    parser.add_argument('--patch_z', type=int, required=False, default=8)
    parser.add_argument('--patch_overlap_x', type=int, required=False, default=0)
    parser.add_argument('--patch_overlap_y', type=int, required=False, default=0)
    parser.add_argument('--patch_overlap_z', type=int, required=False, default=2)
    parser.add_argument('--queue_length', type=int, default=500)
    parser.add_argument('--samples_per_volume', type=int, default=16)

    parser.add_argument('--no_noisy', action='store_true', default=False)
    parser.add_argument('--curriculum', action='store_true', default=True)

    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--continue_epoch', type=int)
    parser.add_argument('--continue_batch_done', type=int)
    parser.add_argument('--continue_cpt_path', type=str)

    parser.add_argument('--test_time_aug', action='store_true', default=False)

    parser.add_argument('--cpt_dir', type=str, default='cpts')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_path', type=str, required=False)

    parser.add_argument('--task_name', type=str, default='')
    parser.add_argument('--task_name_suffix', type=str, default='')

    parser.add_argument('--eps_decay', default=0.995, type=float)
    parser.add_argument('--eps_min', default=0.01, type=float)
    parser.add_argument('--start_decay_epoch', default=0, type=int)

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    train3d(args)
