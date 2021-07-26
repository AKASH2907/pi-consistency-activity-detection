import sys
import os
import cv2
import torch
import time
import random
import imageio
import argparse
import datetime
import numpy as np
# import pandas as pd
# import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss

from tqdm import tqdm
from pylab import savefig
from itertools import cycle
from tensorboardX import SummaryWriter

from models.aug_capsules_ucf101 import CapsNet
from models.pytorch_i3d import InceptionI3d
from datasets.ucf_dataloader import UCF101DataLoader

from utils.losses import SpreadLoss, DiceLoss, IoULoss, weighted_mse_loss
from utils.metrics import get_accuracy, IOU2
# from utils.helpers import measure_pixelwise_uncertainty


#####################################################
'''
Aim:
    - In this code I try to reduce variance temporally pixel-based approach
    
Exp:
    - More weights to uncertain region - Done
    - More weights to certain region - Done
    - Adaptive approach inverse weightage after few epochs - Done
    - We can do linear ramping kind of thing where we increase the weight of one and reduce of the opp - NOT WORTH IT
    - DUMP ADAPTIVE THING - Negative is hurting
    - 


'''
#####################################################

def visualize_clips(rgb_clips, index, filename):
    if rgb_clips.requires_grad==False:
        rgb_clips = rgb_clips.cpu().numpy()
    else:
        rgb_clips = rgb_clips.cpu().detach().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])
    with imageio.get_writer('./uncertain_vis/orig_{}_{:02d}_gt.gif'.format(filename, index), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i]*255).astype(np.uint8)
            writer.append_data(image) 

def aug_visualize_clips(rgb_clips, index, filename):
    if rgb_clips.requires_grad==False:
        rgb_clips = rgb_clips.cpu().numpy()
    else:
        rgb_clips = rgb_clips.cpu().detach().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])
    with imageio.get_writer('./uncertain_vis/aug_{}_{:02d}_gt.gif'.format(filename, index), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i]*255).astype(np.uint8)
            writer.append_data(image) 


def val_model_interface(minibatch, r=0):
    data = minibatch['data'].type(torch.cuda.FloatTensor)
    action = minibatch['action'].cuda()
    segmentation = minibatch['segmentation']

    output, predicted_action, feat, _ = model(data, action)
    
    class_loss, abs_class_loss = criterion_cls(predicted_action, action)
    loss1 = criterion_seg_1(output, segmentation.float().cuda())
    
    seg_loss = loss1
    total_loss =  seg_loss + class_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss)


# def measure_pixelwise_uncertainty(pred):
#     # Calculating mean across multiple MCD forward passes 
#     # batch_mean, batch_variance = 0, 0
#     count = 0
#     batch_variance = torch.zeros_like(pred, dtype=torch.double)
#     # print(batch_variance.dtype)
#     # exit()
#     for zz in range(0, pred.shape[0]):
#         m_temp = pred[zz][0]
#         # clip_variance = torch.zeros((8, 224, 224))
#         clip_variance = torch.zeros_like(pred[0][0], dtype=torch.double)
#         for temp_cnt in range(8):
#             # print(temp_cnt)
#             if temp_cnt-1<0:
#                 temp_var = m_temp[temp_cnt:temp_cnt+2]
#             elif temp_cnt+1>7:
#                 temp_var = m_temp[temp_cnt-1:]
#             else:
#                 # print(temp_cnt-1, temp_cnt, temp_cnt+1, temp_cnt+2)
#                 temp_var = m_temp[temp_cnt-1:temp_cnt+2]

#             # heatmap visualize
#             temp_var = torch.var(temp_var, dim=0, unbiased=False)
#             # temp_vars = np.var(temp_var.cpu().detach().numpy(), axis=0)
#             # print(sum(temp_vars-temp_var))

#             # print(temp_var.shape, int(temp_var.max()*255), int(temp_var.min()*255))
#             # heatmap_img = (temp_var*255).astype(np.uint8)
#             # heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
#             # cv2.imwrite("heatmap_uncertain_vis.png", heatmap_img)
#             # exit()
#             # df_cm = pd.DataFrame((temp_var*255).astype(np.uint8))
#             # svm = sns.heatmap(df_cm, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
#             # figure = svm.get_figure()    
#             # figure.savefig('svm_conf.png', dpi=400)
#             # temp_var = torch.clamp(temp_var, min=0.2, max=1.0)
#             # print(temp_var)
#             # temp_var = 1 - temp_var
#             # print(temp_var)

#             # exit()
#             # clip_variance[temp_cnt] = torch.from_numpy(temp_var)
#             clip_variance[temp_cnt] = temp_var


#         # clip_variance = torch.reshape(clip_variance, (1, clip_variance.shape[0], clip_variance.shape[1], clip_variance.shape[2]))
#         clip_variance = clip_variance.unsqueeze(0)
#         # print(clip_variance.max(), clip_variance.min())
#         # print(temp_var.shape, temp_var.max(), temp_var.min())

#         # this is numpy already change the visualize function
#         # visualize_clips(clip_variance, 0, 'clip_var')
#         # print(clip_variance.shape, type(clip_variance))

#         # clip_variance = torch.from_numpy(clip_variance)
#         # print(clip_variance.max(), clip_variance.min())
#         # print(clip_variance.shape, type(clip_variance))

#         batch_variance[zz] = clip_variance

#         # print(batch_variance.shape, batch_variance.max(), batch_variance.min())
#     # print(type(batch_variance), type(batch_variance[0]))
#     # batch_variance = torch.from_numpy(batch_variance)
#     print(batch_variance.max(), batch_variance.min())
#     print(batch_variance.min(0, keepdim=True)[0].shape)
#     # A -= A.min(1, keepdim=True)[0]
#     # A /= A.max(1, keepdim=True)[0]
#     # batch_variance -= batch_variance.min(1, keepdim=True)[0]
#     # batch_variance /= batch_variance.max(1, keepdim=True)[0]
#     batch_variance -= batch_variance.min()
#     batch_variance /= (batch_variance.max() - batch_variance.min())
#     print(batch_variance.max(), batch_variance.min())


#     exit()

#     # Calculating variance across multiple MCD forward passes 
#     # variance = np.var(pred.cpu().detach().numpy(), axis=0) # shape (n_samples, n_classes)
#     # print(variance.shape)
#     # imageio.imwrite('uncertain_vis/var_8.png', (variance*255).astype(np.uint8))

#     # epsilon = sys.float_info.min
#     # # Calculating entropy across multiple MCD forward passes 
#     # entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

#     # # Calculating mutual information across multiple MCD forward passes 
#     # mutual_info = entropy - np.mean(np.sum(-pred*np.log(pred + epsilon),
#                                             # axis=-1), axis=0) # shape (n_samples,)

#     return batch_variance

def measure_pixelwise_uncertainty(pred, debug=False):
    count = 0
    batch_variance = np.zeros((8, 1, 8, 224, 224))
    # print(batch_variance.dtype)
    for zz in range(0, pred.shape[0]):
        m_temp = pred[zz][0]
        clip_variance = np.zeros((8, 224, 224))
        for temp_cnt in range(8):
            if temp_cnt-1<0:
                temp_var = m_temp[temp_cnt:temp_cnt+2]
            elif temp_cnt+1>7:
                temp_var = m_temp[temp_cnt-1:]
            else:
                temp_var = m_temp[temp_cnt-1:temp_cnt+2]

            # heatmap visualize
            temp_var = np.var(temp_var.cpu().detach().numpy(), axis=0)
            # print(temp_var.max(), temp_var.min())
            temp_var -= temp_var.min()
            temp_var /= (temp_var.max() - temp_var.min())

            #####################################################
            # Adaptive approach to flip after few epochs
            # not giving any boost to simple uncertain so remove it
            # for now later look into it
            #####################################################
            # if epoch<5:
            # temp_var = 1 - temp_var
            # print("normalized", temp_var.max(), temp_var.min())
            #####################################################

            #####################################################
            # VISUALIZE HEAT MAP
            if debug:
                print("Analyze the uncertain regions in heatmaps")
                print(temp_var.shape, int(temp_var.max()*255), int(temp_var.min()*255))
                heatmap_img = (temp_var*255).astype(np.uint8)
                heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                cv2.imwrite("heatmap_uncertain_normalize_neg.png", heatmap_img)
                exit()
            #####################################################

            clip_variance[temp_cnt] = temp_var
        clip_variance = np.reshape(clip_variance, (1, clip_variance.shape[0], clip_variance.shape[1], clip_variance.shape[2]))

        batch_variance[zz] = clip_variance
    # print(batch_variance.min(), batch_variance.max())
    # batch_variance -= batch_variance.min()
    # batch_variance /= (batch_variance.max() - batch_variance.min())
    # print(batch_variance.min(), batch_variance.max())
    # exit()
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance


def measure_pixelwise_uncertainty_v2(pred, debug=False):
    count = 0
    batch_variance = np.zeros_like(pred.cpu().detach().numpy(), dtype=np.float64)
    # print(batch_variance.dtype)
    for zz in range(0, pred.shape[0]):
        m_temp = pred[zz][0]
        clip_variance = np.zeros_like(batch_variance[0][0])
        # print(batch_variance.shape, batch_variance.dtype, clip_variance.shape, clip_variance.dtype)
        for temp_cnt in range(8):
            if temp_cnt-1<0:
                temp_var = m_temp[temp_cnt:temp_cnt+2]
            elif temp_cnt+1>7:
                temp_var = m_temp[temp_cnt-1:]
            else:
                temp_var = m_temp[temp_cnt-1:temp_cnt+2]

            # heatmap visualize
            temp_var = np.var(temp_var.cpu().detach().numpy(), axis=0)
            # print(temp_var.max(), temp_var.min())
            temp_var -= temp_var.min()
            temp_var /= (temp_var.max() - temp_var.min())

            #####################################################
            # Adaptive approach to flip after few epochs
            # not giving any boost to simple uncertain so remove it
            # for now later look into it
            #####################################################
            # if epoch<5:
            # temp_var = 1 - temp_var
            # print("normalized", temp_var.max(), temp_var.min())
            #####################################################

            #####################################################
            # VISUALIZE HEAT MAP
            if debug:
                print("Analyze the uncertain regions in heatmaps")
                print(temp_var.shape, int(temp_var.max()*255), int(temp_var.min()*255))
                heatmap_img = (temp_var*255).astype(np.uint8)
                heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                cv2.imwrite("heatmap_uncertain_normalize_neg.png", heatmap_img)
                exit()
            #####################################################

            clip_variance[temp_cnt] = temp_var
        clip_variance = np.reshape(clip_variance, (1, clip_variance.shape[0], clip_variance.shape[1], clip_variance.shape[2]))

        batch_variance[zz] = clip_variance
    # print(batch_variance.min(), batch_variance.max())
    # batch_variance -= batch_variance.min()
    # batch_variance /= (batch_variance.max() - batch_variance.min())
    # print(batch_variance.min(), batch_variance.max())
    # exit()
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance





def train_model_interface(args, label_minibatch, unlabel_minibatch, epoch):
    '''
    :label_minibatch:


    '''

    label_data = label_minibatch['data'].type(torch.cuda.FloatTensor)
    fl_label_data = label_minibatch['flip_data'].type(torch.cuda.FloatTensor)

    unlabel_data = unlabel_minibatch['data'].type(torch.cuda.FloatTensor)
    fl_unlabel_data = unlabel_minibatch['flip_data'].type(torch.cuda.FloatTensor)

    label_action = label_minibatch['action'].cuda()
    fl_label_action = label_minibatch['action'].cuda()

    unlabel_action = unlabel_minibatch['action'].cuda()
    fl_unlabel_action = unlabel_minibatch['action'].cuda()

    label_segmentation = label_minibatch['segmentation']
    fl_label_segmentation = label_minibatch['flip_label']

    unlabel_segmentation = unlabel_minibatch['segmentation']
    fl_unlabel_segmentation = unlabel_minibatch['flip_label']

    concat_data = torch.cat([label_data, unlabel_data], dim=0)
    concat_fl_data = torch.cat([fl_label_data, fl_unlabel_data], dim=0)
    concat_action = torch.cat([label_action, unlabel_action], dim=0)
    concat_seg = torch.cat([label_segmentation, unlabel_segmentation], dim=0)
    concat_fl_seg = torch.cat([fl_label_segmentation, fl_unlabel_segmentation], dim=0)

    sup_vid_labels = label_minibatch['label_vid']
    unsup_vid_labels = unlabel_minibatch['label_vid']
    concat_labels = torch.cat([sup_vid_labels, unsup_vid_labels], dim=0).cuda()
    random_indices = torch.randperm(len(concat_labels))

    concat_data = concat_data[random_indices, :, :, :, :]
    concat_fl_data = concat_fl_data[random_indices, :, :, :,:]
    concat_action = concat_action[random_indices]
    concat_labels = concat_labels[random_indices]
    concat_seg = concat_seg[random_indices, :, :, :, :]
    concat_fl_seg = concat_fl_seg[random_indices, :, :, :, :]

    labeled_vid_index = torch.where(concat_labels==1)[0]

    output, predicted_action, feat, pen_segmap = model(concat_data, concat_action)
    flip_op, flip_ap, flip_feat, flip_pen_segmap = model(concat_fl_data, concat_action)
    
    # SEG LOSS SUPERVISED
    labeled_op = output[labeled_vid_index]
    labeled_seg_data = concat_seg[labeled_vid_index]
    # print(output.max(), output.min())
    seg_loss_1 = criterion_seg_1(labeled_op, labeled_seg_data.float().cuda())
    seg_loss_2 = criterion_seg_2(labeled_op, labeled_seg_data.float().cuda())
    
    # Classification loss SUPERVISED
    labeled_cls = concat_action[labeled_vid_index]
    labeled_pred_action = predicted_action[labeled_vid_index]
    class_loss, abs_class_loss = criterion_cls(labeled_pred_action, labeled_cls)

    # CONST LOSS
    flipped_pred_seg_map = torch.flip(flip_op, [4])
    flip_pen_segmap = torch.flip(flip_pen_segmap, [4])

    DEBUG = False
    if DEBUG == True:
        visualize_clips(concat_data[0], 0, 'clip')
        visualize_clips(concat_seg[0], 0, 'mask')

    # batch_variance = measure_pixelwise_uncertainty(output)
    # batch_variance = batch_variance.type(torch.cuda.FloatTensor)
    # loss_wt = weighted_mse_loss(flipped_pred_seg_map, output, batch_variance)
    
    # if epoch<6:
    #     equal_wt = torch.ones_like(output, dtype=torch.double)
    #     equal_wt = equal_wt.type(torch.cuda.FloatTensor)
    #     loss_wt = weighted_mse_loss(flipped_pred_seg_map, output, equal_wt)

    # else:

    batch_variance_orig_clip = measure_pixelwise_uncertainty_v2(output)
    batch_variance_orig_clip = batch_variance_orig_clip.type(torch.cuda.FloatTensor)
    loss_wt = weighted_mse_loss(flipped_pred_seg_map, output, batch_variance_orig_clip)

    # batch_variance_aug_clip = measure_pixelwise_uncertainty(flipped_pred_seg_map)
    # batch_variance_aug_clip = batch_variance_aug_clip.type(torch.cuda.FloatTensor)
    # print('cuda device:', multi_batch_variance.dtype)
    #####################################################
    # The idea is that measure the uncertainty temporally and then use it as
    # a weight. No need to take the uncertainty of augmented clip 
    #####################################################
    # print(batch_variance_orig_clip.shape)
    
    # print(equal_wt.shape)

    total_cons_loss = loss_wt    

    seg_loss = seg_loss_1 + seg_loss_2
    total_loss = args.wt_seg * seg_loss + args.wt_cls * class_loss + args.wt_cons * total_cons_loss

    return (output, predicted_action, concat_seg, concat_action, total_loss, seg_loss, class_loss, total_cons_loss)




def train(args, model, labeled_train_loader, unlabeled_train_loader, optimizer, epoch, r, save_path, writer, short=False):
    start_time = time.time()
    steps = len(unlabeled_train_loader)
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    consistency_loss = []
    print('epoch  step    loss   seg  class const  accuracy')
    
    start_time = time.time()
    for batch_id, (label_minibatch, unlabel_minibatch)  in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        if short:
            print("this condition")
            if batch_id > 40:
                break
    
        optimizer.zero_grad()
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = train_model_interface(args, label_minibatch, unlabel_minibatch, epoch)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        consistency_loss.append(cc_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        report_interval = 10
        if (batch_id + 1) % report_interval == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(seg_loss).mean()
            r_class = np.array(class_loss).mean()
            r_cc_class = np.array(class_consistency_loss).mean()
            r_acc = np.array(accuracy).mean()
            print('%d/%d  %d/%d  %.3f  %.3f %.3f %.3f  %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class, r_cc_class, r_acc))

            # summary writing
            total_step = (epoch-1)*len(unlabeled_train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_seg': r_seg,
                'loss_cls': r_class,
                'loss_consistency':r_cc_class
            }
            info_acc = {
            'acc': r_acc
            }

            writer.add_scalars('train/loss', info_loss, total_step)
            writer.add_scalars('train/acc', info_acc, total_step)
            sys.stdout.flush()

    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    # r_total = np.array(total_loss).mean()
    
    #file = 'weights' + str(epoch)
    torch.save(model.state_dict(), save_path+".pth")
    print('saved weights to ', save_path+".pth")
    train_total_loss = np.array(total_loss).mean()
    return r, train_total_loss


def validate(model, val_data_loader, epoch, short=False):
    steps = len(val_data_loader)
    model.eval()
    model.training = False
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()
    
    with torch.no_grad():
        
        for batch_id, minibatch in enumerate(val_data_loader):
            if short:
                if batch_id > 40:
                    break
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss = val_model_interface(minibatch, r)
            total_loss.append(loss.item())
            seg_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))


            maskout = output.cpu()
            maskout_np = maskout.data.numpy()
            # utils.show(maskout_np[0])

            # use threshold to make mask binary
            maskout_np[maskout_np > 0] = 1
            maskout_np[maskout_np < 1] = 0
            # utils.show(maskout_np[0])

            truth_np = segmentation.cpu().data.numpy()
            for a in range(minibatch['data'].shape[0]):
                iou = IOU2(truth_np[a], maskout_np[a])
                if iou == iou:
                    total_IOU += iou
                    validiou += 1
                else:
                    print('bad IOU')
    
    val_epoch_time = time.time() - start_time
    print("Validation time: ", val_epoch_time)
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(seg_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    average_IOU = total_IOU / validiou
    print('Validation %d  %.3f  %.3f  %.3f  %.3f IOU %.3f' % (epoch, r_total, r_seg, r_class, r_acc, average_IOU))
    sys.stdout.flush()
    return r_total



def parse_args():
    parser = argparse.ArgumentParser(description='add_losses')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--pretrained', type=bool, default=True, help='loading pretrained model')
    parser.add_argument('--seg_loss', type=str, default='dice', help='dice or iou loss')
    # parser.add_argument('--log', type=str, default='log_prp', help='log directory')
    parser.add_argument('--exp_id', type=str, default='loss_checks', help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, help='experiment name')
    parser.add_argument('--pkl_file_unlabel', type=str, help='experiment name')
    parser.add_argument('--const_loss', type=str, help='consistency loss type')
    parser.add_argument('--wt_seg', type=float, default=1, help='segmentation loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='Classification loss weight')
    parser.add_argument('--wt_cons', type=float, default=1, help='class consistency loss weight')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    # parser.add_argument('--pretrained', type=bool, )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    USE_CUDA = True if torch.cuda.is_available() else False
    if torch.cuda.is_available() and not USE_CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    TRAIN_BATCH_SIZE = args.bs

    VAL_BATCH_SIZE = args.bs
    N_EPOCHS = args.epochs
    LR = args.lr
    seg_loss_criteria = args.seg_loss

    
    percent = str(100)
    args.pkl_file_label = "train_annots_10_labeled_random.pkl"
    args.pkl_file_unlabel = "train_annots_90_unlabeled_random.pkl"
    # labeled_trainset = UCF101DataLoader('train', [224, 224], TRAIN_BATCH_SIZE, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)
    labeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=4, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)

    unlabeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=12, file_id=args.pkl_file_unlabel, percent=percent, use_random_start_frame=False)

    validationset = UCF101DataLoader('validation',[224, 224], VAL_BATCH_SIZE, file_id="test_annots.pkl", use_random_start_frame=False)
    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))
    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=TRAIN_BATCH_SIZE//2,
        num_workers=8,
        shuffle=True
    )

    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE)//2,
        num_workers=8,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=8,
        shuffle=False
    )

    print(len(labeled_train_data_loader), len(unlabeled_train_data_loader), len(val_data_loader))
    
    # Load pretrained weights
    model = CapsNet(pretrained_load=args.pretrained)
    
    if USE_CUDA:
        model = model.cuda()
    
    # losses
    global criterion_cls
    global criterion_seg_1
    global criterion_seg_2
    global consistency_criterion
    criterion_cls = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion_seg_1 = nn.BCEWithLogitsLoss(size_average=True)

    if seg_loss_criteria == 'dice':
        criterion_seg_2 = DiceLoss()

    if seg_loss_criteria == 'iou':
        criterion_seg_2 = IoULoss()

    if args.const_loss == 'jsd':
        consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    elif args.const_loss == 'l2':
        consistency_criterion = nn.MSELoss()

    elif args.const_loss == 'l1':
        consistency_criterion = nn.L1Loss()

    elif args.const_loss == 'dice':
        consistency_criterion = DiceLoss()

    elif args.const_loss == 'iou':
        consistency_criterion = IoULoss()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1, verbose=True)

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5) # lr is min lr
    # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, cycle_mult=1.0, max_lr=0.001, min_lr=0.000001, warmup_steps=5, gamma=0.1)
    
    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/activity_detect/caps_net/exp_4_data_aug/train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    
    prev_best_val_loss = 10000
    prev_best_train_loss = 10000
    prev_wt_cons_loss = 10000
    prev_best_val_loss_model_path = None
    prev_best_train_loss_model_path = None
    r = 0
    for e in tqdm(range(1, N_EPOCHS + 1)):
        
        r, train_loss = train(args, model, labeled_train_data_loader, unlabeled_train_data_loader, optimizer, e, r, save_path, writer, short=False)

        val_loss = validate(model, val_data_loader, e, short=False)
        if val_loss < prev_best_val_loss:
            print("Yay!!! Got the val loss down...")
            val_model_path = os.path.join(model_save_dir, f'best_model_val_loss_{e}.pth')
            torch.save(model.state_dict(), val_model_path)
            prev_best_val_loss = val_loss;
            if prev_best_val_loss_model_path:
                os.remove(prev_best_val_loss_model_path)
            prev_best_val_loss_model_path = val_model_path

        if train_loss < prev_best_train_loss:
            print("Yay!!! Got the train loss down...")
            train_model_path = os.path.join(model_save_dir, f'best_model_train_loss_{e}.pth')
            torch.save(model.state_dict(), train_model_path)
            prev_best_train_loss = train_loss
            if prev_best_train_loss_model_path:
                os.remove(prev_best_train_loss_model_path)
            prev_best_train_loss_model_path = train_model_path
        scheduler.step(train_loss);

        if e % 20 == 0:
            checkpoints = os.path.join(model_save_dir, f'model_{e}.pth')
            torch.save(model.state_dict(),checkpoints)
            print("save_to:",checkpoints);



'''


def multi_pixelwise_uncertainty(pred, aug_pred):
    # Calculating mean across multiple MCD forward passes 
    # batch_mean, batch_variance = 0, 0
    count = 0

    # visualize_clips(pred[0], 0, 'pred_mask_temp')
    # visualize_clips(aug_pred[0], 0, 'aug_pred_mask_temp')

    aug_pred = torch.flip(aug_pred, [4])
    total_pred = pred+ aug_pred

    # visualize_clips(total_pred[0], 0, 'sum_pred_mask_temp')
    # print(total_pred.shape)

    batch_variance = np.zeros((8, 1, 8, 224, 224))

    # print(batch_variance.shape)
    for zz in range(0, pred.shape[0]):
        # if zz-1<0:

        m_temp = pred[zz][0]
        # print(type(m_temp), m_temp.shape)
        clip_variance = np.zeros((8, 224, 224))
        for temp_cnt in range(8):
            # print(temp_cnt)
            if temp_cnt-1<0:
                temp_var = m_temp[temp_cnt:temp_cnt+2]
            elif temp_cnt+1>7:
                temp_var = m_temp[temp_cnt-1:]
            else:
                temp_var = m_temp[temp_cnt-1:temp_cnt+2]

            # heatmap visualize
            temp_var = np.var(temp_var.cpu().detach().numpy(), axis=0)
            # df_cm = pd.DataFrame((temp_var*255).astype(np.uint8))
            # svm = sns.heatmap(df_cm, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
            # figure = svm.get_figure()    
            # figure.savefig('svm_conf.png', dpi=400)
            # temp_var = np.reshape(temp_var, (1, temp_var.shape[0], temp_var.shape[1], temp_var.shape[2]))
            clip_variance[temp_cnt] = temp_var
        # print(clip_variance.shape)
        clip_variance = np.reshape(clip_variance, (1, clip_variance.shape[0], clip_variance.shape[1], clip_variance.shape[2]))
        # print(clip_variance.shape)
            # print(clip_variance.max(), clip_variance.min())
            # print(temp_var.shape, temp_var.max(), temp_var.min())

        # this is numpy already change the visualize function
        # visualize_clips(clip_variance, 0, 'clip_var')
        # print(clip_variance.shape, type(clip_variance))

        # clip_variance = torch.from_numpy(clip_variance)
        # print(clip_variance.max(), clip_variance.min())
        # print(clip_variance.shape, type(clip_variance))

        batch_variance[zz] = clip_variance
        # print(batch_variance.shape, batch_variance.max(), batch_variance.min())
    # print(type(batch_variance), type(batch_variance[0]))
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance

# mean = np.mean(m_temp.cpu().detach().numpy(), axis=0) # shape (n_samples, n_classes)
# variance = np.var(m_temp.cpu().detach().numpy(), axis=0) # shape (n_samples, n_classes)

# print(mean.shape, variance.shape)
# mean_sum_clip = np.sum(mean)
# var_sum_clip = np.sum(variance)

# print(mean_sum_clip, var_sum_clip)
# batch_mean += mean_sum_clip
# batch_variance += var_sum_clip
# print(count)
# count+=1

elif const_loss == 'dice':
    consistency_criterion = DiceLoss()

    cons_loss_1 = consistency_criterion(flipped_pred_seg_map, output)
    # cons_loss_2 = consistency_criterion(flip_pen_segmap, pen_segmap)
    total_cons_loss = cons_loss_1

elif const_loss == 'iou':
    consistency_criterion = IoULoss()
    cons_loss_1 = consistency_criterion(flipped_pred_seg_map, output)
    # cons_loss_2 = consistency_criterion(flip_pen_segmap, pen_segmap)
    total_cons_loss = cons_loss_1 

elif const_loss == 'l2_dice':
    cc_mse = nn.MSELoss()
    cons_loss_1 = cc_mse(flipped_pred_seg_map, output)

    cc_dice = DiceLoss()
    cons_loss_2 = cc_dice(flipped_pred_seg_map, output)

    total_cons_loss = cons_loss_1 + cons_loss_2

elif const_loss == 'l2_iou':
    cc_mse = nn.MSELoss()
    cons_loss_1 = cc_mse(flipped_pred_seg_map, output)

    cc_iou = IoULoss()
    cons_loss_2 = cc_iou(flipped_pred_seg_map, output)

    total_cons_loss = cons_loss_1 + cons_loss_2


'''
