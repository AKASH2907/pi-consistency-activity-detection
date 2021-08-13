"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.const_losses import LossComputer, iou_hooks_mo, iou_hooks_so
from model import var_wt_losses
from util.log_integrator import Integrator
# from util.image_saver import pool_pairs

import cv2
import imageio
import numpy as np

def visualize_clips(rgb_clips, index, filename):
    rgb_clips = rgb_clips.cpu().detach().numpy()
    # rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])
    rgb_clips = np.transpose(rgb_clips, (0, 2, 3, 1))

    with imageio.get_writer(f'./orig_{filename}_{index:02d}.gif', mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            rgb_clips[i] = (rgb_clips[i] - rgb_clips[i].min())/(rgb_clips[i].max() - rgb_clips[i].min())
            # print(rgb_clips[i].max(), rgb_clips[i].min())
            image = (rgb_clips[i]*255).astype(np.uint8)
            # print(image.max(), image.min())
            writer.append_data(image) 

    writer.close()

def visualize_cls_prediction(gt, index, filename):
    gt = gt.cpu().detach().numpy()
    # gt = np.transpose(gt, (0, 2, 3, 1))

    gt = np.transpose(gt, (1, 2, 0))
    print(gt.shape)

    with imageio.get_writer(f'./gt_{filename}_{index:02d}.gif', mode='I') as writer:
        for i in range(gt.shape[0]):
            # rgb_clips[i] = (rgb_clips[i] - rgb_clips[i].min())/(rgb_clips[i].max() - rgb_clips[i].min())
            # print(rgb_clips[i].max(), rgb_clips[i].min())
            image = (gt[i]*255).astype(np.uint8)
            # print(image.max(), image.min())
            writer.append_data(image) 

    writer.close()

class STCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)
        self.var_loss_computer = var_wt_losses.LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        # self.save_model_interval = 10000
        self.save_model_interval = 5000

        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, label_data, unlabel_data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        data = {}
        # for k, v in label_data.items():
        #     print(k)
        # print(label_data['info'])
        # exit()
        DEBUG = False

        Fsl = label_data['rgb']
        Msl = label_data['gt']
        sec_Msl = label_data['sec_gt']
        selector_l = label_data['selector']
        cls_gt_l = label_data['cls_gt']

        Fsul = unlabel_data['rgb']
        Msul = unlabel_data['gt']
        sec_Msul = unlabel_data['sec_gt']
        selector_ul = unlabel_data['selector']
        cls_gt_ul = unlabel_data['cls_gt']

        data['rgb'] = torch.cat([Fsl, Fsul], axis=0)
        data['gt'] = torch.cat([Msl, Msul], axis=0)
        data['sec_gt'] = torch.cat([sec_Msl, sec_Msul], axis=0)
        data['selector'] = torch.cat([selector_l, selector_ul], axis=0)
        data['cls_gt'] = torch.cat([cls_gt_l, cls_gt_ul], axis=0)


        # if DEBUG:
        #     print(Fsl.shape, Msl.shape, Fsul.shape, Msul.shape)
        #     print(sec_Msl.shape, sec_Msul.shape)
        #     print(selector_l.shape, selector_ul.shape)

        #     print(data['rgb'].shape, data['gt'].shape)
        #     print(data['sec_gt'].shape)
        #     print(data['selector'].shape)


        ones_tensor = torch.ones(Fsl.shape[0])
        zeros_tensor = torch.zeros(Fsul.shape[0])
        concat_labels = torch.cat([ones_tensor, zeros_tensor], dim=0).cuda()
        random_indices = torch.randperm(len(concat_labels))

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        out_fl = {}
        label_data_loss = {}
        labeled_out = {}

        Fs = data['rgb']
        Ms = data['gt']
        # print(Fs.shape)
        # visualize_clips(Fs[0], 0, 'fs_orig')
        Fs = Fs[random_indices, :, :, :, :]
        Ms = Ms[random_indices, :, :, :, :]
        concat_labels = concat_labels[random_indices]
        cls_gt = data['cls_gt']
        cls_gt = cls_gt[random_indices]

        # take Fs flip that, do the same for Ms
        Fs_flip  = torch.flip(Fs, [4])
        Ms_flip = torch.flip(Ms, [4])

        # print(Fs_flip.shape, Ms_flip.shape)

        # visualize_clips(Fs[0], 0, 'fs_noflip')
        # visualize_clips(Fs_flip[0], 0, 'fs_flip')
        # exit()

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)

            k16_fl, kf16_thin_fl, kf16_fl, kf8_fl, kf4_fl = self.STCN('encode_key', Fs_flip)

            sec_Ms = data['sec_gt']
            selector = data['selector']
            sec_Ms = sec_Ms[random_indices, :, :, :, :]
            selector = selector[random_indices, :]
            sec_Ms_flip = torch.flip(sec_Ms, [4])

            ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
            ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
            ref_v = torch.stack([ref_v1, ref_v2], 1)

            ref_v1_fl = self.STCN('encode_value', Fs_flip[:,0], kf16_fl[:,0], Ms_flip[:,0], sec_Ms_flip[:,0])
            ref_v2_fl = self.STCN('encode_value', Fs_flip[:,0], kf16_fl[:,0], sec_Ms_flip[:,0], Ms_flip[:,0])
            ref_v_fl = torch.stack([ref_v1_fl, ref_v2_fl], 1)

            # Segment frame 1 with frame 0
            prev_logits, prev_mask = self.STCN('segment', 
                    k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                    k16[:,:,0:1], ref_v, selector)

            prev_logits_fl, prev_mask_fl = self.STCN('segment', 
                    k16_fl[:,:,1], kf16_thin_fl[:,1], kf8_fl[:,1], kf4_fl[:,1], 
                    k16_fl[:,:,0:1], ref_v_fl, selector)
            
            prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
            prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
            prev_v = torch.stack([prev_v1, prev_v2], 1)
            values = torch.cat([ref_v, prev_v], 3)

            prev_v1_fl = self.STCN('encode_value', Fs_flip[:,1], kf16_fl[:,1], prev_mask_fl[:,0:1], prev_mask_fl[:,1:2])
            prev_v2_fl = self.STCN('encode_value', Fs_flip[:,1], kf16_fl[:,1], prev_mask_fl[:,1:2], prev_mask_fl[:,0:1])
            prev_v_fl = torch.stack([prev_v1_fl, prev_v2_fl], 1)
            values_fl = torch.cat([ref_v_fl, prev_v_fl], 3)

            if DEBUG:
                print(k16.shape, kf16_thin.shape, kf16.shape, kf8.shape, kf4.shape)
                print(ref_v.shape, ref_v1.shape, ref_v2.shape)
                print(prev_logits.shape, prev_mask.shape)
                print(prev_v1.shape, prev_v.shape, values.shape)


                print(k16_fl.shape, kf16_thin_fl.shape, kf16_fl.shape, kf8_fl.shape, kf4_fl.shape)
                print(ref_v_fl.shape, ref_v1_fl.shape, ref_v2_fl.shape)
                print(prev_logits_fl.shape, prev_mask_fl.shape)
                print(prev_v1_fl.shape, prev_v_fl.shape, values_fl.shape)

            del ref_v

            # exit()
            # Segment frame 2 with frame 0 and 1
            this_logits, this_mask = self.STCN('segment', 
                    k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                    k16[:,:,0:2], values, selector)

            this_logits_fl, this_mask_fl = self.STCN('segment', 
                    k16_fl[:,:,2], kf16_thin_fl[:,2], kf8_fl[:,2], kf4_fl[:,2], 
                    k16_fl[:,:,0:2], values_fl, selector)

            out['mask_1'] = prev_mask[:,0:1]
            out['mask_2'] = this_mask[:,0:1]
            out['sec_mask_1'] = prev_mask[:,1:2]
            out['sec_mask_2'] = this_mask[:,1:2]
            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits
            # out['logits'] = torch.cat([prev_logits.unsqueeze(1), this_logits.unsqueeze(1)], dim=1)
            # print(out['logits'].shape)
            # print(prev_mask[8][0])

            # out_fl['mask_1'] = prev_mask_fl[:,0:1]
            # out_fl['mask_2'] = this_mask_fl[:,0:1]
            # out_fl['sec_mask_1'] = prev_mask_fl[:,1:2]
            # out_fl['sec_mask_2'] = this_mask_fl[:,1:2]
            out_fl['logits_1'] = torch.flip(prev_logits_fl, [3])
            out_fl['logits_2'] = torch.flip(this_logits_fl, [3])
            # out_fl['logits'] = torch.cat([out_fl['logits_1'].unsqueeze(1), out_fl['logits_2'].unsqueeze(1)], dim=1)
            # print(out_fl['logits'].shape)
            # exit()

            labeled_vid_index = torch.where(concat_labels==1)[0]
            # print(labeled_vid_index)
            labeled_out['mask_1'] = prev_mask[labeled_vid_index,0:1]
            labeled_out['mask_2'] = this_mask[labeled_vid_index,0:1]
            labeled_out['sec_mask_1'] = prev_mask[labeled_vid_index,1:2]
            labeled_out['sec_mask_2'] = this_mask[labeled_vid_index,1:2]

            labeled_out['logits_1'] = prev_logits[labeled_vid_index]
            labeled_out['logits_2'] = this_logits[labeled_vid_index]

            label_data_loss['rgb'] = Fs[labeled_vid_index]
            label_data_loss['gt'] = Ms[labeled_vid_index]
            label_data_loss['sec_gt'] = sec_Ms[labeled_vid_index]
            label_data_loss['selector'] = selector[labeled_vid_index]
            label_data_loss['cls_gt'] = cls_gt[labeled_vid_index]
            # print(label_data_loss['gt'].shape, label_data_loss['cls_gt'].shape)
            # print(labeled_out['mask_1'].shape, labeled_out['logits_1'].shape)
            # print(torch.equal(label_data_loss['gt'][0][0][0], label_data_loss['cls_gt'][0][0]))  # THEY ARE SAME
            # print(torch.equal(labeled_out['mask_1'][0][0], labeled_out['logits_1'][0][1]))

            # predicted mask and logits are not same
            # print(torch.equal(out['logits_1'][0][0], out_fl['logits_1'][0][0]))

            # exit()
            # print(label_data_loss['rgb'].shape, label_data_loss['cls_gt'].shape)

            # print(out)
            # print(out['mask_1'][0][0])
            

            # print(torch.equal(out['mask_1'][0][0], prev_mask[8][0]))
            # for k, v in out.items():
            #     print(k)
            #     print(out[k].shape)
            # print(out['logits_2'][0].max(), out['logits_2'][0].min())

            # visualize_clips(out['logits_2'], 0, 'log_2')
            # print(label_data_loss['cls_gt'].shape, label_data_loss['gt'].shape)
            # visualize_cls_prediction(label_data_loss['gt'][0], 0, 'seg_gt')
            # visualize_cls_prediction(label_data_loss['cls_gt'][0], 0, 'cls_gt')


            # cls_map = label_data_loss['cls_gt'][0].cpu().detach().numpy()

            # cls_map = np.transpose(cls_map, (1, 2, 0))
            # cv2.imwrite('cls_gt1.png', (cls_map[1]*255).astype(np.uint8))
            # exit()

            # IMP NOTE: logits are nothing but mask of 3 frames placed together

            if self._do_log or self._is_train:
                # losses = self.loss_computer.compute({**data, **out}, it)
                # losses = self.loss_computer.compute({**label_data_loss, **labeled_out}, {**out}, {**out_fl}, it)
                losses = self.var_loss_computer.compute({**label_data_loss, **labeled_out}, {**out}, {**out_fl}, it)

                # print(losses)
                # exit()

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    # if self._is_train:
                    #     if it % self.save_im_interval == 0 and it != 0:
                    #         if self.logger is not None:
                    #             images = {**label_data_loss, **labeled_out}
                    #             size = (384, 384)
                    #             self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # exit()
            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self

