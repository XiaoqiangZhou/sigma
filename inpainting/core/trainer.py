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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist


from inpainting.core.dataset import my_inp_Dataset, my_inp_finetune_Dataset
from inpainting.core.utils import set_seed, set_device, Progbar, postprocess
from inpainting.core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19
import xq_models_mae as models_mae
from xq_utils import get_the_thumbnail_from_grid_masked_image
import cv2
import torchvision.transforms as transforms


class Trainer():
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0


        # setup data set and data loader
        # self.train_dataset = Dataset(
        #     config['data_loader'], debug=debug, split='train')
        # self.train_dataset = my_inp_Dataset(config)
        self.train_dataset = my_inp_finetune_Dataset(config)
        worker_init_fn = partial(set_seed, base=config.seed)
        self.train_sampler = None
        if config.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset,
                                                    num_replicas=config.world_size, rank=config.global_rank)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=config.batch_size // config.world_size,
                                       shuffle=(self.train_sampler is None), num_workers=8,
                                       pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)

        # set up losses and metrics
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = set_device(PerceptualLoss())
        self.style_loss = set_device(StyleLoss())
        self.adversarial_loss = set_device(AdversarialLoss(type="hinge"))
        
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config.global_rank == 0 or (not config.distributed):
            self.dis_writer = SummaryWriter(os.path.join(config.save_dir, 'models', 'dis'))
            self.gen_writer = SummaryWriter(os.path.join(config.save_dir, 'models', 'gen'))

        net = importlib.import_module('inpainting.model.'+config.model_name)
        self.netG = set_device(net.InpaintGenerator(config))
        self.netD = set_device(net.Discriminator(opt=config, use_sigmoid="hinge"!='hinge'))
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=1e-4 * 0.1, betas=(0.5, 0.999))
        if config.use_mae:
            self.net_mae = models_mae.__dict__['mae_vit_large_patch16']()
            print("******* Load pre-trained checkpoint from: %s" % config.mae_model_path)
            checkpoint = torch.load(config.mae_model_path, map_location='cpu')
            self.net_mae.load_state_dict(checkpoint['model'], strict=True)
            self.net_mae = set_device(self.net_mae)
            self.net_mae.eval()
            
        self.load()
        if config.distributed:
            self.netG = DDP(self.netG, device_ids=[config.global_rank], output_device=config.global_rank, broadcast_buffers=True, find_unused_parameters=False)
            self.netD = DDP(self.netD, device_ids=[config.global_rank], output_device=config.global_rank, broadcast_buffers=True, find_unused_parameters=False)
            if self.config.use_mae:
                self.net_mae = DDP(self.net_mae, device_ids=[config.global_rank], output_device=config.global_rank, broadcast_buffers=True, find_unused_parameters=False)

    # get current learning rate
    def get_lr(self, type='G'):
        if type == 'G':
            return self.optimG.param_groups[0]['lr']
        return self.optimD.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        # decay = 0.1**(min(self.iteration, self.config.niter_steady) // self.config.niter)
        decay = 0.1**(min(self.epoch, self.config.nepoch_steady) // self.config.nepoch)
        new_lr = 1e-4 * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # load netG and netD
    def load(self):
        model_path = os.path.join(self.config.save_dir, 'models')
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            print("Loading pretrained models, resume training: ", model_path, latest_epoch)
            gen_path = os.path.join(
                model_path, 'epoch_{}_gen.pth'.format(str(latest_epoch).zfill(3)))
            dis_path = os.path.join(
                model_path, 'epoch_{}_dis.pth'.format(str(latest_epoch).zfill(3)))
            opt_path = os.path.join(
                model_path, 'epoch_{}_opt.pth'.format(str(latest_epoch).zfill(3)))
            if self.config.global_rank == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(
                gen_path, map_location=lambda storage, loc: set_device(storage))
            self.netG.load_state_dict(data['netG'])
            data = torch.load(
                dis_path, map_location=lambda storage, loc: set_device(storage))
            self.netD.load_state_dict(data['netD'])
            data = torch.load(
                opt_path, map_location=lambda storage, loc: set_device(storage))
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config.global_rank == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, step=None):
        model_path = os.path.join(self.config.save_dir, 'models')
        if self.config.global_rank == 0:
            if step is None:
                gen_path = os.path.join(
                    model_path, 'epoch_{}_gen.pth'.format(str(self.epoch).zfill(3)))
                dis_path = os.path.join(
                    model_path, 'epoch_{}_dis.pth'.format(str(self.epoch).zfill(3)))
                opt_path = os.path.join(
                    model_path, 'epoch_{}_opt.pth'.format(str(self.epoch).zfill(3)))
            else:
                gen_path = os.path.join(
                    model_path, 'step_{}_gen.pth'.format(str(self.iteration).zfill(7)))
                dis_path = os.path.join(
                    model_path, 'step_{}_dis.pth'.format(str(self.iteration).zfill(7)))
                opt_path = os.path.join(
                    model_path, 'step_{}_opt.pth'.format(str(self.iteration).zfill(7)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG, netD = self.netG.module, self.netD.module
            else:
                netG, netD = self.netG, self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(self.epoch).zfill(3),
                                            os.path.join(model_path, 'latest.ckpt')))

    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0
    
    def visualize(self, masks, mae_img, pred_img, img, epoch, iteration):
        dataset_mean = torch.tensor(self.train_dataset.dataset_rgb_mean)[None,:,None,None].to(masks.device)
        dataset_std = torch.tensor(self.train_dataset.dataset_rgb_std)[None,:,None,None].to(masks.device)
        images = img * dataset_std + dataset_mean
        pred_img = pred_img * dataset_std + dataset_mean
        # mae_img = F.interpolate(mae_img, (self.config.crop_size, self.config.crop_size))
        mae_img = mae_img * dataset_std + dataset_mean
        masked_img = (images * (1 - masks).float()) + masks
        visual = torch.cat([masked_img, mae_img, pred_img, images], dim=2)
        save_image(visual, os.path.join(self.config.save_dir, 'visual', 'epoch_{}_iter_{}.jpg').format(epoch, iteration))
    
    def vis(self, img):
        dataset_mean = torch.tensor(self.train_dataset.dataset_rgb_mean)[None,:,None,None].to(img.device)
        dataset_std = torch.tensor(self.train_dataset.dataset_rgb_std)[None,:,None,None].to(img.device)
        img = img * dataset_std + dataset_mean
        return img

    def forward_mae_v1(self, images, masks, mask_types): # TODO应该对这里优化一下，第一个if的逻辑也不对
        if mask_types[0] in ['Every_N_Lines', 'Nearest_Neighbor']: # mae cannot handle these two types
            images_masked = (images * (1 - masks).float()) + masks # mask first is necessary since the test time only masked image is available
            return images, masks, images_masked, images_masked, False
        else:# process with mae
            # use the same masks for data in a batch
            masks = torch.unsqueeze(masks[0], 0)
            masks = masks.repeat(images.shape[0], 1, 1, 1) 
            masks_rsz = F.interpolate(masks, (224, 224), mode='nearest')
            # check if the mask appropriate 
            try:
                masks_rsz_patchfied = self.net_mae.patchify(masks_rsz, channel=1)
            except:
                masks_rsz_patchfied = self.net_mae.module.patchify(masks_rsz, channel=1)
            
            if (masks_rsz_patchfied.mean(-1)>self.config.mae_threshold).float().sum(-1)[0]==0 or (masks_rsz_patchfied.mean(-1)>self.config.mae_threshold).float().sum(-1)[0]==196:
                print("skipping data in this iteration, and use last valid in formal iteration")
                images = self.valid_imgs
                masks = self.valid_masks
                masks_rsz = self.valid_masks_rsz
            else:
                self.valid_imgs = images
                self.valid_masks = masks
                self.valid_masks_rsz = masks_rsz
            # to now, the images and masks can be surely be processed by MAE
            images_masked = (images * (1 - masks).float()) + masks # mask first is necessary since the test time only masked image is available
            images_masked_rsz = F.interpolate(images_masked, (224, 224), mode='bicubic', align_corners=True)
            with torch.no_grad():
                reconstruction_loss, pred, mask = self.net_mae(images_masked_rsz, masks_rsz, threshold=self.config.mae_threshold)
                mae_preds = images_masked_rsz * (1 - masks_rsz) + pred * masks_rsz
                mae_preds = F.interpolate(mae_preds, (self.config.crop_size, self.config.crop_size))
            return images, masks, images_masked, mae_preds, True
    
    def get_direct_results(self, ori_img, ori_mask, mae_flag, mask_types):
        img_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.train_dataset.dataset_rgb_mean, std=self.train_dataset.dataset_rgb_std)])
                            
        ori_img = self.vis(ori_img) # batch size should be 1
        ori_img_cv_np = ori_img[0].permute(1,2,0).cpu().numpy()[:,:,::-1]*255.
        ori_mask_cv_np = ori_mask[0].permute(1,2,0).cpu().numpy()[:,:,0]*255.
        try:
            thumbnail = get_the_thumbnail_from_grid_masked_image(ori_img_cv_np, ori_mask_cv_np, return_thumbnail=True)
        except:
            import ipdb;ipdb.set_trace()
        thumbnail = cv2.resize(thumbnail, (ori_img_cv_np.shape[1], ori_img_cv_np.shape[0]))
        thumbnail = Image.fromarray(thumbnail[:,:,::-1].astype(np.uint8))
        thumbnail = img_transform(thumbnail)
        thumbnail = torch.unsqueeze(thumbnail, 0)
        return thumbnail.to(ori_mask.device), torch.ones_like(mae_flag).to(ori_mask.device)

    def forward_mae(self, images, masks, mask_types):
        images_masked = (images * (1 - masks).float()) + masks

        masks_rsz = F.interpolate(masks, (224, 224), mode='nearest')
        try:
            masks_rsz_patchfied = self.net_mae.patchify(masks_rsz, channel=1)
        except:
            masks_rsz_patchfied = self.net_mae.module.patchify(masks_rsz, channel=1)
        
        masked_token_num = (masks_rsz_patchfied.mean(-1)>self.config.mae_threshold).float().sum(-1)

        mae_avail_flag = (masked_token_num!=0) * (masked_token_num!=masks_rsz_patchfied.shape[1])
        mae_avail_flag_tensor = torch.ones_like(masks) * mae_avail_flag[:,None, None, None]
        mae_avail_flag_tensor = mae_avail_flag_tensor.to(images.device)
        
        if mask_types[0] in ['Every_N_Lines', 'Nearest_Neighbor']:
            mae_prediction, mae_avail_flag_tensor = self.get_direct_results(images_masked, masks, mae_avail_flag_tensor, mask_types)
            return images, masks, images_masked, mae_prediction, mae_avail_flag_tensor
        elif mae_avail_flag.sum() == 0: # all samples are unsuitable for mae
            return images, masks, images_masked, images_masked, mae_avail_flag_tensor
        else: # some samples can be processed by mae
            mae_prediction = images_masked.clone()
            images_masked_rsz = F.interpolate(images_masked, (224, 224), mode='bicubic', align_corners=True)
            images_masked_rsz = images_masked_rsz[mae_avail_flag]
            # images_rsz = F.interpolate(images, (224, 224), mode='bicubic', align_corners=True);images_rsz = images_rsz[mae_avail_flag]
            masks_rsz = masks_rsz[mae_avail_flag]
            with torch.no_grad():
                reconstruction_loss, pred, mask = self.net_mae(images_masked_rsz, masks_rsz, threshold=self.config.mae_threshold)
            mae_preds = images_masked_rsz * (1 - masks_rsz) + pred * masks_rsz
            mae_preds = F.interpolate(mae_preds, (self.config.crop_size, self.config.crop_size))
            mae_prediction[mae_avail_flag] = mae_preds # ! important
            return images, masks, images_masked, mae_prediction, mae_avail_flag_tensor

    # process input and calculate loss every training epoch
    def _train_epoch(self):
        progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
        mae = 0
        for images, masks, mask_types in self.train_loader:
            self.iteration += 1
            self.adjust_learning_rate()
            end = time.time()
            # use the same mask for a batch directly!
            
            masks = torch.unsqueeze(masks[0], 0)
            masks = masks.repeat(images.shape[0], 1, 1, 1)

            if self.iteration % 5000 == 0:
                self.save(step=self.iteration)

            if torch.mean(masks)==0:
                print("no missing region")
                continue
            mask_types = [mask_types[0]] * images.shape[0]

            images, masks = set_device([images, masks])
            if self.config.use_mae:
                images, masks, images_masked, mae_preds, mae_flag = self.forward_mae(images, masks, mask_types)
            else:
                images_masked = (images * (1 - masks).float()) + masks 
                mae_preds = images_masked
                mae_flag = None
                # mae_flag = torch.zeros_like(masks).to(masks.device)

            # in: [rgb(3) + edge(1)]
            pred_img = self.netG(masked_img=images_masked, mae_prediction=mae_preds, mask=masks, mae_flag=mae_flag)

            comp_img = (1 - masks)*images + masks*pred_img
            self.add_summary(self.dis_writer, 'lr/dis_lr', self.get_lr(type='D'))
            self.add_summary(self.gen_writer, 'lr/gen_lr', self.get_lr(type='G'))

            gen_loss = 0
            dis_loss = 0
            # image discriminator loss
            dis_real_feat = self.netD(images)
            dis_fake_feat = self.netD(comp_img.detach())
            dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_summary(self.dis_writer, 'loss/dis_fake_loss', dis_fake_loss.item())
            if torch.isnan(dis_loss).sum():
                print("NAN dis loss:", dis_fake_loss, dis_real_loss)
                continue
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # generator adversarial loss
            # in: [rgb(3)]
            gen_fake_feat = self.netD(comp_img)
            gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False)
            gen_loss += gen_fake_loss * self.config.adversarial_weight
            self.add_summary(self.gen_writer, 'loss/gen_fake_loss', gen_fake_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_img*masks, images*masks) / torch.mean(masks)
            gen_loss += hole_loss * self.config.hole_weight
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())
            valid_loss = self.l1_loss(pred_img*(1-masks), images*(1-masks)) / torch.mean(1-masks)
            gen_loss += valid_loss * self.config.valid_weight
            self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
            # generator perceptual loss
            gen_perceptual_loss = self.perceptual_loss(pred_img, images)
            gen_perceptual_loss = gen_perceptual_loss * self.config.perceptual_weight
            self.add_summary(self.gen_writer, 'loss/perceptual_loss', gen_perceptual_loss.item())
            gen_loss += gen_perceptual_loss

            # generator style loss
            gen_style_loss = self.style_loss(pred_img * masks, images * masks)
            gen_style_loss = gen_style_loss * self.config.style_weight
            self.add_summary(self.gen_writer, 'loss/style_loss', gen_style_loss.item())
            gen_loss += gen_style_loss
            
            if torch.isnan(gen_loss).sum():
                print("NAN gen loss:", hole_loss, valid_loss, gen_perceptual_loss, gen_style_loss)
                continue
            
            
            # generator backward
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()
            
                
            # logs
            if torch.mean(masks)!=0:
                new_mae = (torch.mean(torch.abs(images - pred_img)) / torch.mean(masks)).item()  
            else:
                import ipdb;ipdb.set_trace()
            mae = new_mae if mae == 0 else (new_mae+mae)/2
            speed = images.size(0)/(time.time() - end) * self.config.world_size
            logs = [("epoch", self.epoch), ("iter", self.iteration), ("lr", self.get_lr()), ('mae', mae), ('gen_loss', gen_loss.item()), ('samples/s', speed)]
            if self.config.global_rank == 0 and self.iteration%self.config.print_every==0:
                progbar.add(len(images)*self.config.world_size*self.config.print_every, values=logs if 2 else [x for x in logs if not x[0].startswith('l_')])
            
            if (self.iteration+1) % self.config.vis_step == 0 and self.config.global_rank == 0:
                self.visualize(masks, mae_preds, pred_img, images, self.epoch, self.iteration+1)
            
            if self.iteration > self.config.iterations or self.epoch > self.config.epoches:
                break
            
        # saving and evaluating
        if self.epoch % self.config.save_epoch == 0:
            self.save()
        


    def _test_epoch(self, it):
        if self.config.global_rank == 0:
            print('[**] Testing in backend ...')
            model_path = os.path.join(self.config.save_dir, 'models')
            result_path = '{}/results_{}_level_03'.format(
                model_path, str(it).zfill(5))
            log_path = os.path.join(model_path, 'valid.log')
            try:
                os.popen('python test.py -c {} -n {} -l 3 -m {} -s {} > valid.log;'
                         'CUDA_VISIBLE_DEVICES=1 python eval.py -r {} >> {};'
                         'rm -rf {}'.format(self.config['config'], self.config['model_name'], self.config['data_loader']['mask'], self.config['data_loader']['w'],
                                            result_path, log_path, result_path))
            except (BrokenPipeError, IOError):
                pass

    def train(self):
        while True:
            self.epoch += 1
            if self.config.distributed:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch()
            if self.iteration > self.config.iterations or self.epoch > self.config.epoches:
                break
        print('\nEnd training....')
