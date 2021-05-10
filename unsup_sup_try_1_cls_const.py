import sys
import os
import utils
import torch
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from capsules_ucf101 import CapsNet
from itertools import cycle

from torch.utils.data import DataLoader
from torch import optim
import time
import random
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import load_ucf101_pytorch_polate
# from load_ucf101_pytorch_polate import UCF101DataLoader
from ucf_dataloader import UCF101DataLoader

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


def val_model_interface(minibatch, r=0):
    data = minibatch['data']
    flip_data = minibatch['flip_data']
    action = minibatch['action']
    segmentation = minibatch['segmentation']
    segmentation = segmentation
    action = action.cuda()

    data = data.type(torch.cuda.FloatTensor)
    flip_data = flip_data.type(torch.cuda.FloatTensor)

    criterion5 = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion1 = nn.BCEWithLogitsLoss(size_average=True)
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    output, predicted_action, feat = model(data, action)
    flip_op, flip_ap, flip_feat = model(flip_data, action)
    
    class_loss, abs_class_loss = criterion5(predicted_action, action, r)
    loss1 = criterion1(output, segmentation.float().cuda())

    # CONS_LOSS
    predicted_action+= 1e-7
    flip_ap+=1e-7

    cons_loss_a = conf_consistency_criterion(predicted_action.log(), flip_ap.detach()).sum(-1).mean()
    cons_loss_b = conf_consistency_criterion(flip_ap.log(), predicted_action.detach()).sum(-1).mean()
    total_cons_loss = cons_loss_a + cons_loss_b

    
    seg_loss = loss1
    total_loss =  seg_loss + class_loss + total_cons_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss, total_cons_loss)

def train_model_interface(label_minibatch, unlabel_minibatch, r=0):
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

    ones_tensor = torch.ones(len(label_action))
    zeros_tensor = torch.zeros(len(unlabel_action))
    concat_labels = torch.cat([ones_tensor, zeros_tensor], dim=0).cuda()
    random_indices = torch.randperm(len(concat_labels))

    concat_data = concat_data[random_indices, :, :, :, :]
    concat_fl_data = concat_fl_data[random_indices, :, :, :,:]
    concat_action = concat_action[random_indices]
    concat_labels = concat_labels[random_indices]
    concat_seg = concat_seg[random_indices, :, :, :, :]
    concat_fl_seg = concat_fl_seg[random_indices, :, :, :, :]

    # losses
    criterion5 = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion1 = nn.BCEWithLogitsLoss(size_average=True)
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    labeled_vid_index = torch.where(concat_labels==1)[0]

    output, predicted_action, feat = model(concat_data, concat_action)
    flip_op, flip_ap, flip_feat = model(concat_fl_data, concat_action)
    
    # SEG LOSS SUPERVISED
    labeled_op = output[labeled_vid_index]
    labeled_seg_data = concat_seg[labeled_vid_index]
    loss1 = criterion1(labeled_op, labeled_seg_data.float().cuda())
    
    # Classification loss SUP+UNSUP
    labeled_cls = concat_action[labeled_vid_index]
    labeled_pred_action = predicted_action[labeled_vid_index]
    #print(labeled_cls, labeled_pred_action)
    #exit()
    class_loss, abs_class_loss = criterion5(predicted_action, concat_action, r)

    predicted_action+= 1e-7
    flip_ap+=1e-7

    cons_loss_a = conf_consistency_criterion(predicted_action.log(), flip_ap.detach()).sum(-1).mean()
    cons_loss_b = conf_consistency_criterion(flip_ap.log(), predicted_action.detach()).sum(-1).mean()
    # feat = feat + 1e-7
    # flip_feat = flip_feat + 1e-7
    # cons_loss_a = conf_consistency_criterion(feat.log(), flip_feat.detach()).sum(-1).mean()
    # cons_loss_b = conf_consistency_criterion(flip_feat.log(), feat.detach()).sum(-1).mean()
    total_cons_loss = cons_loss_a + cons_loss_b

    seg_loss = loss1
    total_loss = 0.01*seg_loss + class_loss + total_cons_loss
    # print(seg_loss, class_loss, total_cons_loss)
    # exit()

    # return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss)
    return (output, predicted_action, concat_seg, concat_action, total_loss, seg_loss, class_loss, total_cons_loss)




def train(model, labeled_train_loader, unlabeled_train_loader, optimizer, epoch, r, save_path, writer, short=False):
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
        # batch_id = 0
        # minibatch = next(iter(train_loader))
        # print(minibatch)
        # exit()
        # print(batch_id)
        # if batch_id>130:
        #     print(label_minibatch.keys(), unlabel_minibatch.keys())
        optimizer.zero_grad()
        # print(minibatch['segmentation'].shape)
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)
    
        # with torch.cuda.amp.autocast():
        
        # key check 1
        # try:
        #     assert label_minibatch['video_label_avail'].shape==label_minibatch['action'].shape
        # except:
        #     print("shape doesnt match labeled data")
        # output, predicted_action, segmentation, action, loss, s_loss, c_loss = model_interface(minibatch, r)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = train_model_interface(label_minibatch, unlabel_minibatch, r)

        loss.backward()
        # scaler.scale(loss).backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

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


def validate(model, val_data_loader, epoch, short=False):
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
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = val_model_interface(minibatch, r)
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
    # parser.add_argument('--log', type=str, default='log_prp', help='log directory')
    parser.add_argument('--exp_id', type=str, default='loss_checks', help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, help='experiment name')
    parser.add_argument('--pkl_file_unlabel', type=str, help='experiment name')

    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    torch.backends.cudnn.benchmark = True
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
    LR = 0.001
    
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
        batch_size=TRAIN_BATCH_SIZE//4,
        num_workers=8,
        shuffle=True
    )

    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=(3*TRAIN_BATCH_SIZE)//4,
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
    # labeled_batch = next(iter(labeled_train_data_loader))
    # unlabeled_batch = next(iter(unlabeled_train_data_loader))
    # print(labeled_batch['data'].shape, unlabeled_batch['data'].shape)
    # print(labeled_batch['video_label_avail'])
    # print(labeled_batch['action'], unlabeled_batch['action'])
    # print(type(labeled_batch['data']))
    # concat_data = torch.cat([labeled_batch['data'], unlabeled_batch['data']], dim=0)
    # concat_action = torch.cat([labeled_batch['action'], unlabeled_batch['action']], dim=0)
    # print(concat_data.shape, concat_action)
    # print(labeled_batch['segmentation'].shape)
    # exit()
    # print(concat_data[0, 0, 3, :, :])
    # indices = torch.randperm(16)
    # shuffled_concat_data = concat_data[indices, :, :, :, :]
    # shuffled_concat_action = concat_action[indices]
    # print(shuffled_concat_action)
    # print(shuffled_concat_data[0, 0, 3, :, :])
    # exit()

    # counter = 0
    # for item1, item2 in zip(cycle(labeled_train_data_loader), unlabeled_train_data_loader):
    #     counter+=1
        
    # print(counter)
    # exit()

    # pickle files number of annotations check
    # with open("train_annots_80_unlabeled_data.pkl", "rb") as trains:
    #     train_unalbeled_annots = pickle.load(trains)
    # with open("train_annots_20_per.pkl", "rb") as trainssss:
    #     train_labeled_annots = pickle.load(trainssss)
    
    # Load pretrained weights
    model = CapsNet(pretrained_load=True)
    # model.load_previous_weights('best/weights56_427')
    
    if USE_CUDA:
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=50, factor=0.1)

    # training_specs = 'Multi_'+percent+'perEqui_Interp'
    # exp_id = f'i3d-ucf101-{LR}_8_ADAM_capsules_{training_specs}_RGB_Spread_BCE_mixed_precision'
    # exp_id = f'1_partial_train'
    # exp_id = '1_orig_'+ percent
    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/activity_detect/caps_net/exp_4_data_aug/train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # exp_id = os.path.join(save_root)

    
    prev_best_val_loss = 10000
    prev_best_loss_model_path = None
    r = 0
    for e in tqdm(range(1, N_EPOCHS + 1)):
        
        # save_path = os.path.join(save_root, name_prefix)
        # r = train(model, train_data_loader, optimizer, e, r, save_path, writer, short=False)
        r = train(model, labeled_train_data_loader, unlabeled_train_data_loader, optimizer, e, r, save_path, writer, short=False)

        val_loss = validate(model, val_data_loader, e, short=False)
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






















'''
def predict1(idx, wordutil, model, override=''):
    sample = validationset.__getitem__(idx)
    output, predicted_actor, segmentation, actor, loss, loss1, loss2 = model_interface(sample)
    # word_pair = sample['word_pair']
    # print('word_pair',word_pair)
    data = sample['data']
    data = data.view(1, 3, 4, 128, 128)
    sentence = sample['query']
    # if override != "":
    #     sentence = override
    # print('sentence', sentence)
    #
    #
    data = data.type(torch.cuda.FloatTensor)
    # sentence_vec = wordutil.sentences2vec([sentence])
    # segmentation = sample['segmentation']
    # classification = sample['classification']
    # classification = [9]

    # max, prediction = torch.max(avg, 1)
    # print('predicted/actual',prediction,classification)
    maskout = output.cpu()
    maskout_np = maskout.data.numpy()
    # utils.show(maskout_np.reshape(128,128))
    # use threshold to make mask binary
    maskout_np[maskout_np > 0] = 1
    maskout_np[maskout_np < 1] = 0
    maskout_np = maskout_np.reshape(128, 128)
    # utils.show(maskout_np)

    input = data.cpu()
    input_np = input.data.numpy()
    input_np = input_np.reshape(3, 4, 128, 128)

    IOU = utils.IOU(segmentation, maskout_np)
    print('IOU', IOU)
    title = sentence + ' ' + str(IOU)

    # utils.overlay(segmentation, maskout_np, input_np, title)
    utils.side(segmentation, maskout_np, input_np, title)
    utils.byside(segmentation, maskout_np, input_np, title)
    # utils.show(segmentation)
'''
