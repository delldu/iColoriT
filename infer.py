# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import os
import os.path as osp
import pickle
import time
from glob import glob

import torch
import torch.backends.cudnn as cudnn
import torchvision
from einops import rearrange
from timm.models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

import modeling
from datasets import build_fixed_validation_dataset
from utils import lab2rgb, psnr, rgb2lab, seed_worker
import pdb

def get_args():
    parser = argparse.ArgumentParser('Infer Colorization', add_help=False)
    # For evaluation
    parser.add_argument('--model_path', type=str, help='checkpoint path of model', default='models/icolorit_tiny_4ch_patch16_224.pth')
    parser.add_argument('--model_args_path', type=str, help='args.pkl path of model', default='')
    parser.add_argument('--val_data_path', default='data/val/', type=str, help='validation dataset path')
    parser.add_argument('--val_hint_dir', type=str, help='hint directory for fixed validation', default='data/hint')
    parser.add_argument('--pred_dir', type=str, default="output", help='save all prediction here')
    parser.add_argument('--gray_file_list_txt', type=str, default='', help='use gray file list to exclude them')
    parser.add_argument('--return_name', action='store_true', help='return name for saving (True for test)')
    parser.add_argument('--no_return_name', action='store_false', dest='return_name', help='')
    parser.set_defaults(return_name=True)

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='icolorit_tiny_4ch_patch16_224', type=str, help='Name of model to inference')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode', choices=['linear', 'cnn', 'locattn'])
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Hint generator parameter
    parser.add_argument('--hint_size', default=2, type=int, help='size of the hint region is given by (h, h)')
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--val_hint_list', default=[0, 1, 2, 5, 10, 20, 50, 100, 200], nargs='+')

    args = parser.parse_args()

    if osp.isdir(args.model_path):
        all_checkpoints = glob(osp.join(args.model_path, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.model_path = os.path.join(args.model_path, f'checkpoint-{latest_ckpt}.pth')
    print(f'Load checkpoint: {args.model_path}')

    if args.model_args_path: # args.pkl
        with open(args.model_args_path, 'rb') as f:
            train_args = vars(pickle.load(f))
            model_keys = ['model', 'use_rpb', 'head_mode', 'drop_path', 'mask_cent', 'avg_hint']
            for key in model_keys:
                if key in train_args.keys():
                    setattr(args, key, train_args[key])
                else:
                    print(f'{key} is not in {args.model_args_path}. Please check the args.pkl')
            time.sleep(3)
    print(f'Load args: {args.model_args_path}')

    args.val_hint_list = [int(h) for h in args.val_hint_list]
    for count in args.val_hint_list:
        os.makedirs(osp.join(args.pred_dir, f'h{args.hint_size}-n{count}'), exist_ok=True)

    return args


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model, # icolorit_base_4ch_patch16_224
        pretrained=False,
        drop_path_rate=args.drop_path, # 0
        drop_block_rate=None,
        use_rpb=args.use_rpb, # True -- relative positional bias
        avg_hint=args.avg_hint, # True -- average hint
        head_mode=args.head_mode, # cnn
        mask_cent=args.mask_cent,
    )
    # model -- icolorit_base_4ch_patch16_224, IColoriT

    # checkpoint = torch.load("models/icolorit_base_4ch_patch16_224.pth")
    # checkpoint = torch.load("models/icolorit_small_4ch_patch16_224.pth")
    checkpoint = torch.load("models/icolorit_tiny_4ch_patch16_224.pth")
    # Namespace(
    #     model='colorit_tiny_4ch_patch16_224', 
    #     mask_root_dir='2022ECCV_colorization/ctest10k_mask/1234', 
    #     num_hint_range=[0, 128], hint_size=2, 
    #     input_size=224, drop_path=0.0, 
    #     masked_position_generator='RandomMaskingGenerator', 
    #     avg_hint=True, use_rpb=True, blk_mode='default', head_mode='cnn', 
    #     mask_cent=False, mask_n_q=4, mask_mode='uniform', mask_r=4, mask_p=0.125, 
    #     data_path='ImageNet2012/train', 
    #     val_data_path='2022ECCV_colorization/ctest10k', 
    #     imagenet_default_mean_and_std=True, val_hint_list=[0, 10, 100], 
    #     output_dir='./output/colorit_tiny_4ch_patch16_224/rpb_cnnhead', 
    #     mask_dirs=['2022ECCV_colorization/ctest10k_mask/1234/h2:0',
    #         '2022ECCV_colorization/ctest10k_mask/1234/h2:10',
    #         '2022ECCV_colorization/ctest10k_mask/1234/h2:100'],
    #     window_size=(14, 14), patch_size=(16, 16))

    return model


def main(args):
    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    psnr_sum = dict(zip(args.val_hint_list, [0.] * len(args.val_hint_list)))
    total_shown = 0

    args.hint_dirs = [osp.join(args.val_hint_dir, f'h{args.hint_size}-n{i}') for i in args.val_hint_list]

    # (Pdb) pp args
    # Namespace(model_path='models/icolorit_base_4ch_patch16_224.pth', 
    #     model_args_path='', val_data_path='data/val/', val_hint_dir='data/hint', 
    #     pred_dir='output', gray_file_list_txt='', return_name=True, 
    #     input_size=224, batch_size=32, num_workers=4, pin_mem=True, 
    #     model='icolorit_base_4ch_patch16_224', use_rpb=True, 
    #     head_mode='cnn', drop_path=0.0, mask_cent=False, device='cuda', 
    #     hint_size=2, avg_hint=True, val_hint_list=[0, 1, 2, 5, 10, 20, 50, 100, 200], 
    #     window_size=(14, 14), patch_size=(16, 16), 
    #     hint_dirs=['data/hint/h2-n0', 'data/hint/h2-n1', 'data/hint/h2-n2', 
    #     'data/hint/h2-n5', 'data/hint/h2-n10', 'data/hint/h2-n20', 'data/hint/h2-n50', 
    #     'data/hint/h2-n100', 'data/hint/h2-n200'])


    dataset_val = build_fixed_validation_dataset(args)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=seed_worker,
        shuffle=False,
    )

    with torch.no_grad():
        pbar = tqdm(desc=f'Evaluate', ncols=100, total=len(data_loader_val) * len(args.val_hint_list))
        for step, batch in enumerate(data_loader_val):
            (images, bool_hints), targets, names = batch

            # images size: torch.Size([10, 3, 224, 224]), max: 1.0, min: 0.0
            # bool_hints size: torch.Size([10, 9, 112, 112]), max: 1.0, min: 0.0
            # targets size: torch.Size([10]), max: 0, min: 0
            # patch_size: (16, 16)

            B, _, H, W = images.shape
            h, w = H // patch_size[0], W // patch_size[1]

            # batch preparation
            images = images.to(device, non_blocking=True)
            images_lab = rgb2lab(images)


            images_patch = rearrange(images_lab, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                     p1=patch_size[0], p2=patch_size[1])
            labels = rearrange(images_patch, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size[0], p2=patch_size[1])

            # pp args.val_hint_list -- [0, 1, 2, 5, 10, 20, 50, 100, 200]
            # bool_hints.size() -- torch.Size([10, 9, 112, 112])
            # bool_hints.max(), bool_hints.min(), bool_hints.mean()
            # (tensor(1.), tensor(0.), tensor(0.9966))

            for i, count in enumerate(args.val_hint_list):
                bool_hint = bool_hints[:, i]
                bool_hint = bool_hint.to(device, non_blocking=True).flatten(1).to(torch.bool)

                with torch.cuda.amp.autocast():
                    outputs = model(images_lab.clone(), bool_hint.clone())
                    outputs = rearrange(outputs, 'b n (p1 p2 c) -> b n (p1 p2) c', p1=patch_size[0], p2=patch_size[1])

                pred_imgs_lab = torch.cat((labels[:, :, :, 0].unsqueeze(3), outputs), dim=3)
                pred_imgs_lab = rearrange(pred_imgs_lab, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                                          h=h, w=w, p1=patch_size[0], p2=patch_size[1])
                pred_imgs = lab2rgb(pred_imgs_lab)

                psnr_sum[count] += psnr(images, pred_imgs).item() * B

                if args.pred_dir is not None:
                    img_save_dir = osp.join(args.pred_dir, f'h{args.hint_size}-n{count}')
                    for name, pred_img in zip(names, pred_imgs):
                        torchvision.utils.save_image(pred_img.unsqueeze(0), osp.join(
                            img_save_dir, osp.splitext(name)[0] + '.png'))
                pbar.update()
            total_shown += B
            pbar.set_postfix({'psnr@10': psnr_sum.get(10) / total_shown})
        pbar.close()

    print(f'Total shown: {total_shown}')
    print(f'PSNR {10}: {psnr_sum[10]/total_shown}')



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    main(args)
