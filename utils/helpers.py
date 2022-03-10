import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


def measure_pixelwise_var_v2(pred, flip_pred, frames_cnt=5, use_sig_output=False):
    """cyclic variance
    varv3 - cyclic version

    """
    count = 0
    batch_variance = np.zeros((pred.shape[0], 1, 8, 224, 224))

    # remove the redundant frames 1dt n last of flipped map
    temp_batch_var = np.zeros((pred.shape[0], 1, 14, 224, 224))

    # use sigmoid output not logits
    # probability scores
    if use_sig_output==True:
        pred = torch.sigmoid(pred)
        flip_pred = torch.sigmoid(flip_pred)

    for zz in range(0, pred.shape[0]):
        clip = pred[zz][0]
        flip_clip = flip_pred[zz][0]

        cyclic_clip = torch.cat([clip, flip_clip[1:7]], axis=0).cpu().detach().numpy()

        # calculated variance over 14 frames
        clip_variance = np.zeros_like(temp_batch_var[0][0])
        for temp_cnt in range(temp_batch_var.shape[2]):
            # 3 frames
            if frames_cnt==3:
                if temp_cnt+1>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-1, temp_cnt, 0], axis=0)
                else:
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-1, temp_cnt, temp_cnt+1], axis=0)
            # 5 frames
            if frames_cnt==5:
                if temp_cnt+1>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, 0, 1], axis=0)
                elif temp_cnt+2>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, temp_cnt+1, 0], axis=0)
                else:
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, temp_cnt+1, temp_cnt+2], axis=0)
            
            temp_var = np.var(temp_var, axis=0)
            clip_variance[temp_cnt] = temp_var

        # overlap
        for add_half in range(8):
            if add_half==0 or add_half==7:
                clip_variance[add_half] = 2* clip_variance[add_half]
            else:
                clip_variance[add_half] = clip_variance[add_half] + clip_variance[14-add_half]
        # normalize
        clip_variance = clip_variance[:8]
        clip_variance -= clip_variance.min()
        clip_variance /= (clip_variance.max() - clip_variance.min() + 1e-7)
        
        clip_variance = np.expand_dims(clip_variance, axis=0)
        batch_variance[zz] = clip_variance
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance


def measure_pixelwise_gradient(pred, conf_thresh_lower=None, conf_thresh_upper=None):
    """This version uses numpy for creating empty tensors dtype=float64
    This is performing the best keep this for now

    """
    count = 0
    batch_gradient = np.zeros((pred.shape[0], 8, 224, 224))

    pred_sigmoid = torch.sigmoid(pred)

    for zz in range(0, pred.shape[0]):
        pred_clip = pred_sigmoid[zz][0]
        if conf_thresh_lower is not None:
            pred_clip[pred_clip<conf_thresh_lower] = 0
        if conf_thresh_upper is not None:
            pred_clip[pred_clip>conf_thresh_upper] = 1

        clip_gradient = np.gradient(np.gradient(pred_clip.cpu().detach().numpy(), axis=0), axis=0)
        clip_gradient -= clip_gradient.min()
        clip_gradient /= (clip_gradient.max() - clip_gradient.min() + 1e-7)
        
        batch_gradient[zz] = clip_gradient

    batch_gradient = torch.from_numpy(batch_gradient)

    return batch_gradient
