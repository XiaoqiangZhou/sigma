import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from einops import rearrange

def get_grid_mask_type(tiny_mask):
    mean_0 = tiny_mask.mean(0)
    mean_1 = tiny_mask.mean(1)
    mask_h = tiny_mask.shape[0]
    mask_w = tiny_mask.shape[1]
    # TODO: get the down_scaled img
    if len(np.where(mean_0==255)[0])<2 and len(np.where(mean_1==255)[0])<2:
        return None, 0
    elif len(np.where(mean_0!=255)[0])<2 and len(np.where(mean_1!=255)[0])<2:
        return None, 0
    elif 255 in mean_0 and 255 in mean_1:
        mask_type = 'nearest_neighbor'
        how_many_lines = (np.where(mean_0!=255)[0][1] - np.where(mean_0!=255)[0][0])
        if how_many_lines==1: 
            return None, 0
        start_0 = np.where(mean_0!=255)[0][0]
        start_1 = np.where(mean_1!=255)[0][0]
        refill_mask = np.ones((mask_h, mask_w)) * 255
        refill_mask[start_1::how_many_lines, start_0::how_many_lines] = 0
        if (tiny_mask == refill_mask).all():
            return mask_type, [how_many_lines, start_0, start_1]
        else:
            return None, 0
    elif 255 in mean_0 and 255 not in mean_1:
        mask_type = 'vertical'
        how_many_lines = (np.where(mean_0==255)[0][1] - np.where(mean_0==255)[0][0])
        start_0 = np.where(mean_0==255)[0][0]
        refill_mask = np.zeros((mask_h, mask_w))
        refill_mask[:, start_0::how_many_lines] = 255
        if (tiny_mask == refill_mask).all():
            return mask_type, [how_many_lines, start_0]
        else:
            return None, 0
    elif 255 not in mean_0 and 255 in mean_1:
        mask_type= 'horizontal'
        how_many_lines = (np.where(mean_1==255)[0][1] - np.where(mean_1==255)[0][0])
        start_0 = np.where(mean_1==255)[0][0]
        refill_mask = np.zeros((mask_h, mask_w))
        refill_mask[start_0::how_many_lines] = 255
        if (tiny_mask == refill_mask).all():
            return mask_type, [how_many_lines, start_0]
        else:
            return None, 0
    else:
        return None, 0

def get_the_thumbnail_from_grid_masked_image(img, mask, return_thumbnail=True):
    # place this function in the dataloader part
    img_h, img_w, img_c = img.shape
    mask_type, param = get_grid_mask_type(mask[:30, :30])
    if mask_type is not None:
        if mask_type == 'nearest_neighbor':
            how_many_lines, start_0, start_1 = param
            refill_mask = np.ones((img_h, img_w)) * 255
            refill_mask[start_1::how_many_lines, start_0::how_many_lines] = 0
            # assert (mask*1.==refill_mask).all()
            refill_mask = np.repeat(refill_mask[:,:,None], 3, axis=-1)
            target_h = len(np.arange(img_h)[start_1::how_many_lines])
            target_w = len(np.arange(img_w)[start_0::how_many_lines])
        elif mask_type == 'vertical':
            param, start_0 = param
            refill_mask = np.zeros((img_h, img_w))
            refill_mask[:, start_0::param] = 255
            # assert (mask*1.==refill_mask).all()
            refill_mask = np.repeat(refill_mask[:,:,None], 3, axis=-1)
            target_h = img_h
            target_w = img_w - len(np.arange(img_w)[start_0::param])
        elif mask_type == 'horizontal':
            param, start_0 = param
            refill_mask = np.zeros((img_h, img_w))
            refill_mask[start_0::param] = 255
            # assert (mask*1.==refill_mask).all()
            target_h = img_h - len(np.arange(img_h)[start_0::param])
            target_w = img_w
            refill_mask = np.repeat(refill_mask[:,:,None], 3, axis=-1)
        thumbnail = img[refill_mask==0].reshape(target_h, target_w, img_c)
    else:
        return None
    
    if return_thumbnail:
        return thumbnail
    