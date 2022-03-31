import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

class my_inp_test_Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, test_folder, test_mask_type=None):
        super(my_inp_test_Dataset, self).__init__()
        self.opt = opt
        self.test_folder = test_folder
        self.test_mask_type = test_mask_type
        self.masked_imgs, self.masks, self.masked_img_paths, self.mask_paths = self.load_all_test_set(self.test_folder)
        
        self.size = opt.crop_size

        self.mask_types = ['Every_N_Lines', 'Completion', 'Expand', 'Nearest_Neighbor', 'ThickStrokes', 'MediumStrokes', 'ThinStrokes']
    
        if 'places2' in test_folder or 'Places' in test_folder:
            self.dataset_name = 'places2'
            dataset_rgb_mean = [0.458, 0.441, 0.408]
            dataset_rgb_std = [0.239, 0.236, 0.245]
        elif 'ffhq' in test_folder or 'FFHQ' in test_folder: 
            self.dataset_name = 'ffhq'  
            dataset_rgb_mean = [0.520, 0.425, 0.380]
            dataset_rgb_std = [0.253, 0.228, 0.225]
        elif 'imagenet' in test_folder or 'ImageNet' in test_folder:
            self.dataset_name = 'imagenet'
            dataset_rgb_mean = [0.485, 0.456, 0.406]
            dataset_rgb_std = [0.229, 0.256, 0.225]
        elif 'wikiart' in test_folder or 'WikiArt' in test_folder:
            self.dataset_name = 'wikiart'
            dataset_rgb_mean = [0.522, 0.468, 0.407]
            dataset_rgb_std = [0.222, 0.210, 0.198]
        else:
            raise NotImplementedError()
        print("************** Dateset name: {} **************".format(self.dataset_name))

        self.dataset_rgb_mean = dataset_rgb_mean
        self.dataset_rgb_std = dataset_rgb_std

        # simple augmentation
        img_transform = [] if opt.full_size_test else [transforms.Resize((self.size, self.size), interpolation=3)]
        img_transform += [transforms.ToTensor(), transforms.Normalize(mean=dataset_rgb_mean, std=dataset_rgb_std)]
        self.img_transform_test = transforms.Compose(img_transform) # ImageNet mean and std

        mask_transform = [] if opt.full_size_test else [transforms.Resize((self.size, self.size), interpolation=0)]
        mask_transform += [transforms.ToTensor()]
        self.mask_transform_test = transforms.Compose(mask_transform)

    def __len__(self):
        return len(self.masked_img_paths)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item
    
    def load_item(self, index):
        masked_img = self.masked_imgs[index]
        mask = self.masks[index]
        mask_path = self.mask_paths[index]
        flags = np.array([x in mask_path for x in self.mask_paths])
        assert self.test_mask_type != ''
        mask_type = self.test_mask_type
        masked_img_path = self.masked_img_paths[index]
        

        masked_img = self.img_transform_test(masked_img)
        mask = self.mask_transform_test(mask)
        
        return masked_img, mask, mask_type, masked_img_path

    def load_name(self, index):
        name = self.img_paths[index]
        return os.path.basename(name)
    
    def load_all_test_set(self, test_folder_root):
        if self.test_mask_type != '':
            assert self.test_mask_type in test_folder_root
            return self.load_one_test_set(test_folder_root)
        else:
            total_masked_imgs = []
            total_masks = []
            total_masked_img_paths = []
            total_mask_paths = []
            mask_types = os.listdir(test_folder_root)
            for i, mask_type in enumerate(mask_types):
                masked_imgs, masks, masked_img_paths, mask_paths = \
                            self.load_one_test_set(os.path.join(test_folder_root, mask_type))
                total_masked_imgs += masked_imgs    
                total_masks += masks
                total_masked_img_paths += masked_img_paths
                total_mask_paths += mask_paths
            return total_masked_imgs, total_masks, total_masked_img_paths, total_mask_paths

    def load_one_test_set(self, test_folder_path):
        # load masked images and masks
        names = os.listdir(test_folder_path)
        names.sort()
        masked_img_paths = []
        mask_paths = []
        masked_imgs = []
        masks = []
        for i, name in enumerate(names):
            if '_mask' in name or '_segm' in name:
                continue
            else:
                masked_img_path = os.path.join(test_folder_path, name)
                masked_img_paths.append(masked_img_path)
                assert len(name.split('.')) == 2
                mask_name = name.split('.')[0] + '_mask.' + name.split('.')[-1]
                mask_path = os.path.join(test_folder_path, mask_name)
                assert os.path.exists(mask_path), "The mask file may not exist: "+mask_path
                mask_paths.append(mask_path)
                
                try:
                    temp = Image.open(masked_img_path)
                    bb=temp.copy()
                    temp = bb.convert("RGB")
                except:
                    import ipdb;ipdb.set_trace()
                    print(masked_img_path)
                keep = temp.copy()
                masked_imgs.append(keep)
                temp.close()

                temp = Image.open(mask_path)
                keep = temp.copy()
                masks.append(keep)
                temp.close()
        return masked_imgs, masks, masked_img_paths, mask_paths 


