import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F



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

    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)
