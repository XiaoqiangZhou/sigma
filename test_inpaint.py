import sys
import os
import cv2
import time
import requests
import argparse

from PIL import Image

import os
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms

import models_mae as models_mae

from inpainting.core.utils import set_device
from inpainting.core.dataset import my_inp_test_Dataset
from thop import profile
import time
def get_args_parser():
    parser = argparse.ArgumentParser()
    # data setting 
    
    parser.add_argument('--crop_size', type=int, default=256, help='')
    parser.add_argument('--aspect_ratio', type=float, default=1, help='')
    parser.add_argument('--full_size_test', action='store_true', default=False, help='if specified, use the full image size without resize to crop_size')
    
    parser.add_argument('--semantic_nc', type=int, default=4, help='')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--label_nc', type=int, default=0, help='# of input label classes ')

    # mae testing
    parser.add_argument('--mae_model_path', default='*.pth', type=str)
    parser.add_argument('--dataset_name', default='places2', type=str)
    parser.add_argument('--test_root', default='', type=str)
    parser.add_argument('--output_root', default='results/lama_and_mae/', type=str)
    parser.add_argument('--mae_threshold', default=0, type=float)
    # lama testing
    parser.add_argument('--inp_model_name', default='lama', type=str)
    parser.add_argument('--inp_model_path', default='', type=str)
    parser.add_argument('--use_mae', action='store_true', default=False, help='if specified, use the mae prediction as a guidance')
    # some test time manual setting
    parser.add_argument('--replace_background', action='store_true', help='replace the pixels of the known region in the prediction with gt image')
    parser.add_argument('--resize_to_ori', action='store_true', help='resize the test image from 512 to original image size')
    parser.add_argument('--pad', action='store_true', help='padding the image size to make (size/8=integers), to adapt the lama')
    args = parser.parse_args()
    return args

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model

def prepare_inp_model(config):
    net = importlib.import_module('inpainting.model.'+config.inp_model_name)
    netG = set_device(net.InpaintGenerator(config))
    if config.inp_model_path != '':
        print("Load the trained inpainting model from ", config.inp_model_path)
        data = torch.load(config.inp_model_path, map_location='cuda:0')
        netG.load_state_dict(data['netG'], strict=True)
    else:
        print("Warning: There is no pretrained inpainting model being loaded")
    return netG

def get_dataset_mean_and_std(dataset_name):
    if dataset_name == 'places':
        dataset_mean = np.array([0.458, 0.441, 0.408])
        dataset_std = np.array([0.239, 0.236, 0.245])
    elif dataset_name == 'ffhq':
        dataset_mean = np.array([0.520, 0.425, 0.380])
        dataset_std = np.array([0.253, 0.228, 0.225]) 
    elif dataset_name == 'imagenet':
        dataset_mean = np.array([0.485, 0.456, 0.406])
        dataset_std = np.array([0.229, 0.224, 0.225])
    elif dataset_name == 'wikiart':
        dataset_mean = [0.522, 0.468, 0.407]
        dataset_std = [0.222, 0.210, 0.198]
    return dataset_mean, dataset_std


def run_mae_inference(model_mae, images, masks, args):
    images_masked = (images * (1 - masks).float()) + masks
    masks_rsz = F.interpolate(masks, (224, 224), mode='nearest')
    # check if the mask appropriate 

    masks_rsz_patchfied = model_mae.patchify(masks_rsz, channel=1)
    
    masked_token_num = (masks_rsz_patchfied.mean(-1)>args.mae_threshold).float().sum(-1)
    mae_avail_flag = (masked_token_num!=0) * (masked_token_num!=masks_rsz_patchfied.shape[1])
    mae_avail_flag_tensor = torch.ones_like(masks) * mae_avail_flag[:,None, None, None]
    mae_avail_flag_tensor = mae_avail_flag_tensor.to(images.device)

    if mae_avail_flag.sum() == 0:
        from utils import get_the_thumbnail_from_grid_masked_image
        img_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=args.dataset_mean, std=args.dataset_std)])
        ori_img = vis_an_img(images, args)
        ori_img_cv_np = ori_img[0].permute(1,2,0).cpu().numpy()[:,:,::-1]*255.
        ori_mask_cv_np = masks[0].permute(1,2,0).cpu().numpy()[:,:,0]*255.
        thumbnail = get_the_thumbnail_from_grid_masked_image(ori_img_cv_np, ori_mask_cv_np, return_thumbnail=True)
        if thumbnail is None:
            return images_masked, mae_avail_flag_tensor
        thumbnail = cv2.resize(thumbnail, (ori_img_cv_np.shape[1], ori_img_cv_np.shape[0]))
        thumbnail = Image.fromarray(thumbnail[:,:,::-1].astype(np.uint8))
        thumbnail = img_transform(thumbnail)
        thumbnail = torch.unsqueeze(thumbnail, 0)
        return thumbnail.to(masks.device), torch.ones_like(mae_avail_flag_tensor).to(masks.device)
    else:
        # import ipdb;ipdb.set_trace()
        images_masked_rsz = F.interpolate(images_masked, (224, 224), mode='bicubic', align_corners=True)
        with torch.no_grad():
            reconstruction_loss, pred, mask = model_mae(images_masked_rsz, masks_rsz, threshold=args.mae_threshold)
            mae_preds = images_masked_rsz * (1 - masks_rsz) + pred * masks_rsz
            h,w = images.shape[2:]
            mae_preds = F.interpolate(mae_preds, (h, w))
        return mae_preds, mae_avail_flag_tensor

def data_postprocess(ori_img, ori_mask, mae_pred, lama_pred, concat, args):
    mae_pred = mae_pred.cpu()
    lama_pred = lama_pred.cpu()
    dataset_mean = torch.tensor(args.dataset_mean)[None,:,None,None].to(lama_pred.device)
    dataset_std = torch.tensor(args.dataset_std)[None,:,None,None].to(lama_pred.device)
    mae_pred_img = torch.clip((mae_pred * dataset_std + dataset_mean) * 255, 0, 255).int()
    lama_pred_img = torch.clip((lama_pred * dataset_std + dataset_mean) * 255, 0, 255).int()
    
    if not concat:
        _, _, img_h, img_w = ori_img.shape
        ori_img = torch.clip((ori_img.cpu() * dataset_std + dataset_mean) * 255, 0, 255)[0].permute(1,2,0).numpy()[:,:,::-1]
        ori_mask = ori_mask.cpu()[0].permute(1,2,0).numpy()
        lama_pred_img = lama_pred_img[0].permute(1,2,0).cpu().numpy()[:,:,::-1]
        lama_pred_img = cv2.resize(lama_pred_img.astype(np.uint8), (img_w, img_h))
        lama_pred_img = lama_pred_img * ori_mask + ori_img * (1 - ori_mask) if args.replace_background else lama_pred_img
        return lama_pred_img


def vis_an_img(input, args):
    dataset_mean = torch.tensor(args.dataset_mean)[None,:,None,None].to(input.device)
    dataset_std = torch.tensor(args.dataset_std)[None,:,None,None].to(input.device)
    input = (input * dataset_std + dataset_mean)
    return input

def normal_padding(masked_img, mask):
    N, C, H, W = masked_img.shape
    pad_h = H % 8
    pad_w = W % 8
    p = (0, pad_w, 0, pad_h)
    masked_img = F.pad(masked_img, p, 'replicate')
    mask = F.pad(mask, p, 'replicate')
    return masked_img, mask, pad_h, pad_w

def test(model_mae_gan, model_inpaint, test_folder, output_folder, mask_type, args, concat=False):
    test_dataset = my_inp_test_Dataset(args, test_folder, mask_type)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False, num_workers=4)
    
    for i, data in enumerate(test_dataloader):
        masked_img, mask, _, masked_img_path = data
        if args.pad:
            masked_img, mask, pad_h, pad_w = normal_padding(masked_img, mask)
        # preprocess img
        ori_img = masked_img
        ori_mask = mask
        masked_img = (masked_img * (1 - mask).float()) + mask
        masked_img = masked_img.cuda()
        mask = mask.cuda()
        
        # mae prediction
        mae_pred, mae_flag = run_mae_inference(model_mae_gan, masked_img, mask, args)
        
        # lama inpainting prediction
        with torch.no_grad():
            lama_pred = model_inpaint(masked_img=masked_img, mae_prediction=mae_pred, mask=mask, mae_flag=mae_flag)

        out_image = data_postprocess(ori_img, ori_mask, mae_pred, lama_pred, concat, args)
        
        if args.pad:
            if pad_h>0:
                out_image = out_image[:-pad_h, : , :]
            if pad_w>0:
                out_image = out_image[:, :-pad_w, :]

        out_file = os.path.join(output_root, '/'.join(masked_img_path[0].split('/')[-4:]))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        cv2.imwrite(out_file, out_image.astype(np.uint8))
        

if __name__=='__main__':
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    args = get_args_parser()
    assert args.dataset_name in args.test_root.lower()
    dataset_mean, dataset_std = get_dataset_mean_and_std(args.dataset_name)
    args.dataset_mean = dataset_mean
    args.dataset_std = dataset_std

    # load mae model
    # chkpt_dir = 'released_models/mae_visualize_vit_large_ganloss.pth'
    model_mae_gan = prepare_model(args.mae_model_path, 'mae_vit_large_patch16')
    model_mae_gan = model_mae_gan.cuda()
    model_mae_gan.eval()
    print('MAE model loaded.')

    # load inpainting model
    model_inpaint = prepare_inp_model(args)
    # model_inpaint.eval() # It hearts the performance greatly, maybe related with Lama network
    print("Inpaint model loaded.")

    # import ipdb;ipdb.set_trace()
    # size = 512
    # flops, param = profile(model_mae_gan, inputs=(torch.randn(1,3,224,224).cuda(), torch.randn(1,1,224,224).cuda()))
    # print("GFlops:{}, Parma: {} M".format(flops/1e9, param/1e6)) # 35.14 GFlops, 329.12 M (512)  # 36.35 GFlops, 329.12M (256)
    # flops, param = profile(model_inpaint, inputs=(torch.randn(1,3,size,size).cuda(), torch.randn(1,3,size,size).cuda(), torch.randn(1,1,size,size).cuda(), torch.randn(1,1,size,size).cuda()))
    # print("GFlops:{}, Parma: {} M".format(flops/1e9, param/1e6)) # 174.38 GFlops, 27.06 M (512)  # 43.64 GFlops, 27.06M  (256) 

    # test
    test_root = args.test_root
    mask_types = os.listdir(test_root)
    mask_types.sort()
    # mask_types = ['Nearest_Neighbor']
    test_folders = [os.path.join(test_root, _) for _ in mask_types]
    # output_root = os.path.join(args.output_root, args.inp_model_path.split('/')[-1].split('.')[0])
    for i, test_folder in enumerate(test_folders):
        mask_type = test_folder.split('/')[-1]
        output_folder = os.path.join(output_root, mask_type)
        print("Processing folder: ", test_folder)
        test(model_mae_gan, model_inpaint, test_folder, output_folder, mask_type, args, concat=False)