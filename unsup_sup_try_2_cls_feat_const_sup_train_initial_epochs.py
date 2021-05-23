import sys
import os
import utils
import torch
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from models.capsules_ucf101 import CapsNet
from itertools import cycle

from torch.utils.data import DataLoader
from torch import optim
import time
import random
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
from models.pytorch_i3d import InceptionI3d
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

# import load_ucf101_pytorch_polate
# from load_ucf101_pytorch_polate import UCF101DataLoader
from datasets.ucf_dataloader import UCF101DataLoader

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=24):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        target = target.long()
        # target comes in as class number like 23
        # x comes in as a length 64 vector of averages of all locations
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min) * r
        # print('predictions', x[0])
        # print('target',target[0])
        # print('margin',margin)
        # print('target',target.size())

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
            # print('an at value',x[i][lb])
        at = at.view(b, 1).repeat(1, E)
        # print('at shape',at.shape)
        # print('at',at[0])

        zeros = x.new_zeros(x.shape)
        # print('zero shape',zeros.shape)
        absloss = torch.max(.9 - (at - x), zeros)
        loss = torch.max(margin - (at - x), zeros)
        # print('loss',loss.shape)
        # print('loss',loss)
        absloss = absloss ** 2
        loss = loss ** 2
        absloss = absloss.sum() / b - .9 ** 2
        loss = loss.sum() / b - margin ** 2
        loss = loss.sum()/b

        return loss, absloss


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, labels, classes):
        # print('labels',labels[0])
        # print('predictions',classes[0])
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss


def get_accuracy(predicted_actor, actor):
    maxm, prediction = torch.max(predicted_actor, 1)
    prediction = prediction.view(-1, 1)
    actor = actor.view(-1, 1)
    correct = torch.sum(actor == prediction.float()).item()
    accuracy = correct / float(prediction.shape[0])
    return accuracy


def get_accuracy2(predicted_actor, actor):
    # This gets the f-measure of our network
    predictions = predicted_actor > 0.5

    tp = ((predictions + actor) > 1).sum()
    tn = ((predictions + actor) < 1).sum()
    fp = (predictions > actor).sum()
    fn = (predictions < actor).sum()

    return (tp + tn) / (tp + tn + fp + fn)


def val_model_interface(minibatch, wt_seg, wt_cons_cls, const_loss, seg_criterion, class_criterion, consistency_criterion, r=0):
    data = minibatch['data']
    flip_data = minibatch['flip_data']
    action = minibatch['action']
    segmentation = minibatch['segmentation']
    segmentation = segmentation
    action = action.cuda()

    data = data.type(torch.cuda.FloatTensor)
    flip_data = flip_data.type(torch.cuda.FloatTensor)

    output, predicted_action, feat = model(data, action)
    flip_op, flip_ap, flip_feat = model(flip_data, action)
    
    class_loss, abs_class_loss = class_criterion(predicted_action, action, r)
    loss1 = seg_criterion(output, segmentation.float().cuda())

    # CONS_LOSS
    if const_loss == 'jsd':
        consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
        feat += 1e-7
        flip_feat += 1e-7
        cons_loss_a = consistency_criterion(feat.log(), flip_feat.detach()).sum(-1).mean()
        cons_loss_b = consistency_criterion(flip_feat.log(), feat.detach()).sum(-1).mean()
        total_cons_loss = cons_loss_a + cons_loss_b

    elif const_loss == 'l2':
        consistency_criterion = nn.MSELoss()
        total_cons_loss = consistency_criterion(feat, flip_feat)

    elif const_loss == 'l1':
        consistency_criterion = nn.L1Loss()
        total_cons_loss = consistency_criterion(feat, flip_feat)


    
    seg_loss = loss1
    total_loss =  wt_seg * seg_loss + class_loss + wt_cons_cls * total_cons_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss, total_cons_loss)

def train_model_interface(label_minibatch, unlabel_minibatch, wt_seg, wt_cons_cls, const_loss, seg_criterion, class_criterion, consistency_criterion, r=0):
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

    output, predicted_action, feat = model(concat_data, concat_action)
    flip_op, flip_ap, flip_feat = model(concat_fl_data, concat_action)
    
    # SEG LOSS SUPERVISED
    labeled_op = output[labeled_vid_index]
    labeled_seg_data = concat_seg[labeled_vid_index]
    loss1 = seg_criterion(labeled_op, labeled_seg_data.float().cuda())
    
    # Classification loss SUPERVISED
    labeled_cls = concat_action[labeled_vid_index]
    labeled_pred_action = predicted_action[labeled_vid_index]
    class_loss, abs_class_loss = class_criterion(labeled_pred_action, labeled_cls, r)

    if const_loss == 'jsd':
        
        feat += 1e-7
        flip_feat += 1e-7
        cons_loss_a = consistency_criterion(feat.log(), flip_feat.detach()).sum(-1).mean()
        cons_loss_b = consistency_criterion(flip_feat.log(), feat.detach()).sum(-1).mean()
        total_cons_loss = cons_loss_a + cons_loss_b

    elif const_loss == 'l2':
        total_cons_loss = consistency_criterion(feat, flip_feat)

    elif const_loss == 'l1':
        total_cons_loss = consistency_criterion(feat, flip_feat)
    
    seg_loss = loss1
    total_loss = wt_seg * seg_loss + class_loss + wt_cons_cls * total_cons_loss
    return (output, predicted_action, concat_seg, concat_action, total_loss, seg_loss, class_loss, total_cons_loss)

def model_interface(minibatch, wt_seg, wt_cons_cls, const_loss, seg_criterion, class_criterion, consistency_criterion, r=0):
    data = minibatch['data']
    flip_data = minibatch['flip_data']
    action = minibatch['action']
    segmentation = minibatch['segmentation']
    segmentation = segmentation
    action = action.cuda()

    data = data.type(torch.cuda.FloatTensor)
    flip_data = flip_data.type(torch.cuda.FloatTensor)

    output, predicted_action, feat = model(data, action)
    flip_op, flip_ap, flip_feat = model(flip_data, action)
    
    # SUPERVISED LOSSES
    class_loss, abs_class_loss = class_criterion(predicted_action, action, r)
    loss1 = seg_criterion(output, segmentation.float().cuda())

    # CONSISTENCY_LOSS
    if const_loss == 'jsd':
        
        feat += 1e-7
        flip_feat += 1e-7
        cons_loss_a = consistency_criterion(feat.log(), flip_feat.detach()).sum(-1).mean()
        cons_loss_b = consistency_criterion(flip_feat.log(), feat.detach()).sum(-1).mean()
        total_cons_loss = cons_loss_a + cons_loss_b

    elif const_loss == 'l2':
        total_cons_loss = consistency_criterion(feat, flip_feat)

    elif const_loss == 'l1':
        total_cons_loss = consistency_criterion(feat, flip_feat)

    
    seg_loss = loss1
    total_loss = wt_seg * seg_loss + class_loss + wt_cons_cls * total_cons_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss, total_cons_loss)



def init_train(model, labeled_train_loader, optimizer, epoch, r, save_path, writer, wt_seg, wt_cons_cls, const_loss, 
    seg_criterion, class_criterion, consistency_criterion, short=False):
    start_time = time.time()
    steps = len(labeled_train_loader)
    # print('training: batch size ',TRAIN_BATCH_SIZE,' ',N_EPOCHS,'epochs', steps,' steps ')
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_loss_sent = []
    class_consistency_loss = []
    accuracy_sent = []
    print('epoch  step    loss   seg    class consistency  accuracy')
    #optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=0, eps=1e-6)
    
    start_time = time.time()
    for batch_id, label_minibatch  in enumerate(labeled_train_loader):
        if short:
            print("this condition")
            if batch_id > 40:
                break
    
        optimizer.zero_grad()
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = model_interface(label_minibatch, wt_seg, wt_cons_cls, 
                                                                                                        const_loss, seg_criterion,
                                                                                                        class_criterion, consistency_criterion, r)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        class_consistency_loss.append(cc_loss.item())
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
            total_step = (epoch-1)*len(labeled_train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_seg': r_seg,
                'loss_cls': r_class,
                'loss_cls_consistency':r_cc_class
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
    
    #file = 'weights' + str(epoch)
    torch.save(model.state_dict(), save_path+".pth")
    print('saved weights to ', save_path+".pth")
    # print(r)
    # exit()
    return r




def train(model, labeled_train_loader, unlabeled_train_loader, optimizer, epoch, r, save_path, writer, wt_seg, wt_cons_cls, const_loss, 
    seg_criterion, class_criterion, consistency_criterion, short=False):
    start_time = time.time()
    steps = len(unlabeled_train_loader)
    # print('training: batch size ',TRAIN_BATCH_SIZE,' ',N_EPOCHS,'epochs', steps,' steps ')
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_loss_sent = []
    class_consistency_loss = []
    accuracy_sent = []
    print('epoch  step    loss   seg    class consistency  accuracy')
    #optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=0, eps=1e-6)
    
    start_time = time.time()
    for batch_id, (label_minibatch, unlabel_minibatch)  in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        if short:
            print("this condition")
            if batch_id > 40:
                break
        
        optimizer.zero_grad()
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = train_model_interface(label_minibatch, unlabel_minibatch, wt_seg, wt_cons_cls, 
                                                                                                                const_loss, seg_criterion,
                                                                                                                class_criterion, consistency_criterion,
                                                                                                                  r)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        class_consistency_loss.append(cc_loss.item())
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
                'loss_cls_consistency':r_cc_class
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
    
    #file = 'weights' + str(epoch)
    torch.save(model.state_dict(), save_path+".pth")
    print('saved weights to ', save_path+".pth")
    return r


def validate(model, val_data_loader, epoch, wt_seg, wt_cons_cls, const_loss, short=False):
    steps = len(val_data_loader)
    # print('validation: batch size ', VAL_BATCH_SIZE, ' ', N_EPOCHS, 'epochs', steps, ' steps ')
    model.eval()
    model.training = False
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_consistency_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()
    
    with torch.no_grad():
        
        for batch_id, minibatch in enumerate(val_data_loader):
            if short:
                if batch_id > 40:
                    break
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = val_model_interface(minibatch, wt_seg, wt_cons_cls, const_loss, 
                                                                                                                seg_criterion, class_criterion, consistency_criterion, r)
            total_loss.append(loss.item())
            seg_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            class_consistency_loss.append(cc_loss.item())
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
                iou = utils.IOU2(truth_np[a], maskout_np[a])
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
    r_cc_class = np.array(class_consistency_loss).mean()
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
    # parser.add_argument('--log', type=str, default='log_prp', help='log directory')
    parser.add_argument('--exp_id', type=str, default='loss_checks', help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, help='experiment name')
    parser.add_argument('--pkl_file_unlabel', type=str, help='experiment name')
    parser.add_argument('--const_loss', type=str, help='consistency loss type')
    parser.add_argument('--wt_seg', type=float, default=1, help='segmentation loss weight')
    parser.add_argument('--wt_cons_cls', type=float, default=1, help='class consistency loss weight')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')

    parser.add_argument('--lbl_per_btch', type=float, default=0.5, help='percentage of labeled data in a batch')
    parser.add_argument('--unlbl_per_btch', type=float, default=0.5, help='percentage of unlabeled data in a batch')
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

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

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
    LABEL_DATA_PER_BATCH = args.lbl_per_btch
    UNLABEL_DATA_PER_BATCH = args.unlbl_per_btch
    
    percent = str(100)
    args.pkl_file_label = "train_annots_20_labeled.pkl"
    args.pkl_file_unlabel = "train_annots_80_unlabeled.pkl"
    # labeled_trainset = UCF101DataLoader('train', [224, 224], TRAIN_BATCH_SIZE, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)

    labeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=4, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)

    unlabeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=12, file_id=args.pkl_file_unlabel, percent=percent, use_random_start_frame=False)

    validationset = UCF101DataLoader('validation',[224, 224], VAL_BATCH_SIZE, file_id="test_annots.pkl", use_random_start_frame=False)
    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))
    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=int(TRAIN_BATCH_SIZE*LABEL_DATA_PER_BATCH),
        num_workers=8,
        shuffle=True
    )

    init_labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=8,
        shuffle=True
    )

    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=int(TRAIN_BATCH_SIZE*UNLABEL_DATA_PER_BATCH),
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
    # losses
    class_criterion = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    seg_criterion = nn.BCEWithLogitsLoss(size_average=True)

    if args.const_loss == "jsd":
        consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
    elif args.const_loss == "l2":
        consistency_criterion = nn.MSELoss()
    elif args.const_loss == "l1":
        consistency_criterion = nn.L1Loss()
    else:
        print("Hey there!!! You forgot to put consistency loss criterion.")

    
    
    # Load pretrained weights
    model = CapsNet(pretrained_load=True)
    # model.load_previous_weights('best/weights56_427')
    
    if USE_CUDA:
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1, verbose=True)

    # LOGS AND WEIGHTS SAVE PATH
    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/activity_detect/caps_net/exp_4_data_aug/train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    
    prev_best_val_loss = 10000
    prev_best_loss_model_path = None
    r = 0
    for e in tqdm(range(1, N_EPOCHS + 1)):
        
        # save_path = os.path.join(save_root, name_prefix)
        # r = train(model, train_data_loader, optimizer, e, r, save_path, writer, short=False)
        if e <4:
            r = init_train(model, init_labeled_train_data_loader, optimizer, e, r, save_path, writer, 
                args.wt_seg, args.wt_cons_cls, args.const_loss, 
                seg_criterion, class_criterion, consistency_criterion,
                short=False)
        else:

            r = train(model, labeled_train_data_loader, unlabeled_train_data_loader, optimizer, e, r, save_path, writer, 
                args.wt_seg, args.wt_cons_cls, args.const_loss, 
                seg_criterion, class_criterion, consistency_criterion,
                short=False)

        val_loss = validate(model, val_data_loader, e, args.wt_seg, args.wt_cons_cls, args.const_loss, short=False)
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(model_save_dir, f'best_model_{e}.pth')
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss;
            if prev_best_loss_model_path:
                os.remove(prev_best_loss_model_path)
            prev_best_loss_model_path = model_path
        # scheduler.step(val_loss);

        if e % 20 == 0:
            checkpoints = os.path.join(model_save_dir, f'model_{e}.pth')
            torch.save(model.state_dict(),checkpoints)
            print("save_to:",checkpoints);
