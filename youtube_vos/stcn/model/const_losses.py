import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu

from collections import defaultdict

'''
This script calculates the nonweighted L2 consistency weight
between the two logits predicted with different skip rates

'''

def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets, weight):
        return (weight * (input - target) ** 2).mean()

class NonWeightedMSELoss(nn.Module):
    """docstring for NonWeightedMSELoss"""
    def __init__(self):
        super(NonWeightedMSELoss, self).__init__()
        # self.arg = arg

    def forward(self, inputs, targets):
        return F.mse_loss(inputs, targets)
        


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        print(para)
        self.bce = BootstrappedCE()
        self.nonwt_mseloss = NonWeightedMSELoss()

    def compute(self, data, temp_data1, temp_data2, it):
        losses = defaultdict(int)
        # print(data)
        # for k,v in data.items():
        #     print(k)
        #     print(data[k].shape)
        # for k,v in temp_data1.items():
        #     print(k, temp_data1[k].shape)
        # for k,v in temp_data2.items():
        #     print(k, temp_data2[k].shape)
        b, s, _, _, _ = data['gt'].shape   #b - batch_size, s - n_frames
        # print(b, s)
        selector = data.get('selector', None)

        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                else:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)

                # print(loss.is_cuda, loss.dtype())
                losses['loss_%d'%i] += loss / b
                losses['p'] += p / b / (s-1)


            losses['const'] += self.nonwt_mseloss(temp_data1['logits_%d'%i], temp_data2['logits_%d'%i])
            # print(losses['const'])
            losses['total_loss'] += losses['loss_%d'%i] 

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i]>0.5, data['gt'][:,i]>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

        losses['total_loss'] += 0.1 * losses['const']

        return losses
