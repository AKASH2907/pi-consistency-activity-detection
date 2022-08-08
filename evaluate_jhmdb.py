import sys
import os
import glob
import utils
import torch
import random
import cv2
import time
import argparse
import datetime
import numpy as np
import os.path as osp
from pathlib import Path
from shutil import copy2

import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.capsules_jhmdb_semi_sup_pa import CapsNet

def iou():
    """
    Calculates the accuracy, f-mAP, and v-mAP over the test set
    """

    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = CapsNet().cuda()

    
    n_classes = 21
    clip_batch_size = 14
    model_names = list()
    fmap_best = list()
    vmap_best = list()
    wt_count = 0
    filtered_files = [file for file in os.listdir(args.ckpt) if file.endswith(".pth")]

    for saved_wts in sorted(glob.glob(osp.join(args.ckpt, 'best_model_*.pth'))):
        model.load_previous_weights(saved_wts)

        model_names.append(saved_wts)
        model.eval()
        model.training = False

    
        with torch.no_grad():
            from datasets.jhmdb_dataloader_eval import JHMDB
            validationset = JHMDB('test',[224, 224], 1, use_random_start_frame=False)
            val_data_loader = DataLoader(
                dataset=validationset,
                batch_size=1,
                num_workers=4,
                shuffle=False
            )
            
            n_correct, n_vids, n_tot_frames = 0, np.zeros((n_classes, 1)), np.zeros((n_classes, 1))

            frame_ious = np.zeros((n_classes, 20))
            video_ious = np.zeros((n_classes, 20))
            iou_threshs = np.arange(0, 20, dtype=np.float32)/20

            for idx, sample in enumerate(val_data_loader):
                video, bbox, label, vid_name = sample
                video = video[0]
                bbox = bbox[0]
                label = label[0]
                vid_name = vid_name[0]

                f_skip = 2
                clips = []
                n_frames = video.shape[0]
                for i in range(0, video.shape[0], 8*f_skip):
                    for j in range(f_skip):
                        b_vid, b_bbox = [], []
                        for k in range(8):
                            ind = i + j + k*f_skip
                            if ind >= n_frames:
                                b_vid.append(np.zeros((1, 224, 224, 3), dtype=np.float32))
                                b_bbox.append(np.zeros((1, 224, 224, 1), dtype=np.float32))
                            else:
                                b_vid.append(video[ind:ind+1, :, :, :])
                                b_bbox.append(bbox[ind:ind+1, :, :, :])

                        clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                        if np.sum(clips[-1][1]) == 0:
                            clips.pop(-1)

                if len(clips) == 0:
                    print('Video has no bounding boxes')
                    continue

                batches, gt_segmentations = [], []
                for i in range(0, len(clips), clip_batch_size):
                    x_batch, bb_batch, y_batch = [], [], []
                    for j in range(i, min(i+clip_batch_size, len(clips))):
                        x, bb, y = clips[j]
                        x_batch.append(x)
                        bb_batch.append(bb)
                        y_batch.append(y)
                    batches.append((x_batch, bb_batch, y_batch))
                    gt_segmentations.append(np.stack(bb_batch))

                gt_segmentations = np.concatenate(gt_segmentations, axis=0)
                gt_segmentations = gt_segmentations.reshape((-1, 224, 224, 1))  # Shape N_FRAMES, 112, 112, 1

                segmentations, predictions = [], []
                for x_batch, bb_batch, y_batch in batches:
                    data = np.transpose(np.array(x_batch), [0, 4, 1, 2, 3])
                    data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                    empty_action = np.ones((len(x_batch),1),np.int)*500
                    empty_action = torch.from_numpy(empty_action).cuda()
                    
                    segmentation, pred, _ = model(data, empty_action, empty_action, 0, 0)
                    segmentation = F.sigmoid(segmentation)
                    segmentation_np = segmentation.cpu().data.numpy()   # B x C x F x H x W -> B x 1 x 8 x 224 x 224
                    segmentation_np = np.transpose(segmentation_np, [0, 2, 3, 4, 1])    
                    segmentations.append(segmentation_np)
                    predictions.append(pred.cpu().data.numpy())

                predictions = np.concatenate(predictions, axis=0)
                #predictions = predictions.reshape((-1, n_classes))
                assert predictions.shape[1] == n_classes
                fin_pred = np.mean(predictions, axis=0)

                fin_pred = np.argmax(fin_pred)
                if fin_pred == label:

                    n_correct += 1
                    correct_pred.write(vid_name + ' ' + str(fin_pred) + ' ' + str(label.item()) + '\n')
                else:
                    
                    incorrect_pred.write(vid_name + ' ' + str(fin_pred) + ' ' + str(label.item()) + '\n')

                pred_segmentations = np.concatenate(segmentations, axis=0)
                pred_segmentations = pred_segmentations.reshape((-1, 224, 224, 1))

                pred_segmentations = (pred_segmentations >= 0.5).astype(np.int64)
                seg_plus_gt = pred_segmentations + gt_segmentations

                vid_inter, vid_union = 0, 0
                # calculates f_map
                for i in range(gt_segmentations.shape[0]):
                    frame_gt = gt_segmentations[i]
                    if np.sum(frame_gt) == 0:
                        continue

                    n_tot_frames[label] += 1

                    inter = np.count_nonzero(seg_plus_gt[i] == 2)
                    union = np.count_nonzero(seg_plus_gt[i])
                    vid_inter += inter
                    vid_union += union

                    i_over_u = inter / union
                    for k in range(iou_threshs.shape[0]):
                        if i_over_u >= iou_threshs[k]:
                            frame_ious[label, k] += 1

                n_vids[label] += 1
                i_over_u = vid_inter / vid_union
                for k in range(iou_threshs.shape[0]):
                    if i_over_u >= iou_threshs[k]:
                        video_ious[label, k] += 1

            fAP = frame_ious/n_tot_frames
            fmAP = np.mean(fAP, axis=0)
            vAP = video_ious/n_vids
            vmAP = np.mean(vAP, axis=0)

            print('Accuracy:', n_correct / np.sum(n_vids) , iou_threshs[4], fmAP[4], vmAP[4], iou_threshs[10], fmAP[10], vmAP[10])
            fmap_best.append(fmAP[10])
            vmap_best.append(vmAP[10])


    best_fmap_model = model_names[fmap_best.index(max(fmap_best))]
    best_vmap_model = model_names[vmap_best.index(max(vmap_best))]
    best_files = list()
    best_files.append(best_fmap_model)
    best_files.append(best_vmap_model)
    print(os.listdir(args.ckpt))

iou()
