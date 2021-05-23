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
import imageio
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
from models.pytorch_i3d import InceptionI3d
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from train_utils import SpreadLoss, get_accuracy
from datasets.ucf_dataloader_aug_mod_1 import UCF101DataLoader



def val_model_interface(minibatch, r=0):
    data = minibatch['data'].type(torch.cuda.FloatTensor)
    action = minibatch['action'].cuda()
    segmentation = minibatch['segmentation']

    criterion5 = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion1 = nn.BCEWithLogitsLoss(size_average=True)

    output, predicted_action, feat = model(data, action)
    
    class_loss, abs_class_loss = criterion5(predicted_action, action, r)
    loss1 = criterion1(output, segmentation.float().cuda())
    
    seg_loss = loss1
    total_loss =  seg_loss + class_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss)


def visualize_clips(rgb_clips, index, filename):
    rgb_clips = rgb_clips.cpu().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])
    with imageio.get_writer('./dataloader_outputs/orig_{}_{:02d}_gt.gif'.format(filename, index), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i]*255).astype(np.uint8)
            writer.append_data(image) 

def aug_visualize_clips(rgb_clips, index, filename):
    if rgb_clips.requires_grad==False:
        rgb_clips = rgb_clips.cpu().numpy()
    else:
        rgb_clips = rgb_clips.cpu().detach().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])
    with imageio.get_writer('./dataloader_outputs/aug_{}_{:02d}_gt.gif'.format(filename, index), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i]*255).astype(np.uint8)
            writer.append_data(image) 


def train_model_interface(label_minibatch, unlabel_minibatch, wt_seg, wt_cls, wt_cons_cls, const_loss, r=0):
    label_data = label_minibatch['data'].type(torch.cuda.FloatTensor)
    unlabel_data = unlabel_minibatch['data'].type(torch.cuda.FloatTensor)

    label_action = label_minibatch['action'].cuda()
    unlabel_action = unlabel_minibatch['action'].cuda()

    label_segmentation = label_minibatch['segmentation']
    unlabel_segmentation = unlabel_minibatch['segmentation']

    concat_data = torch.cat([label_data, unlabel_data], dim=0)
    concat_action = torch.cat([label_action, unlabel_action], dim=0)
    concat_seg = torch.cat([label_segmentation, unlabel_segmentation], dim=0)

    sup_vid_labels = label_minibatch['label_vid']
    unsup_vid_labels = unlabel_minibatch['label_vid']
    concat_labels = torch.cat([sup_vid_labels, unsup_vid_labels], dim=0).cuda()
    random_indices = torch.randperm(len(concat_labels))

    # RANDOM SHUFFLING
    concat_data = concat_data[random_indices, :, :, :, :]
    concat_action = concat_action[random_indices]
    concat_labels = concat_labels[random_indices]
    concat_seg = concat_seg[random_indices, :, :, :, :]

    # losses
    criterion5 = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion1 = nn.BCEWithLogitsLoss(size_average=True)

    labeled_vid_index = torch.where(concat_labels==1)[0]


    # IMPLEMENT RANDOM AUG HERE GET THE GT HERE VISUALIZE IT
    random_aug = torch.randint(1, 4, (1,))
    # print(random_aug )
    DEBUG = False
    if DEBUG == True:
        visualize_clips(concat_data[0], 0, 'clip')
        visualize_clips(concat_seg[0], 0, 'mask')

    if random_aug == 1:
        # FLIP LR
        aug_data = torch.flip(concat_data, [4])
        aug_mask = torch.flip(concat_seg, [4])

        if DEBUG==True:
            aug_visualize_clips(aug_data[0], 0, 'flipping')
            aug_visualize_clips(aug_mask[0], 0, 'flip_mask')

    elif random_aug == 2:
        # COUNTER CLOCK ROTATION
        aug_data = torch.rot90(concat_data, 1, [3, 4])
        aug_mask = torch.rot90(concat_seg, 1, [3, 4])

        if DEBUG==True:
            visualize_clips(concat_seg[0], 0, 'mask')
            aug_visualize_clips(aug_mask[0], 0, 'counter_clock_mask')


    elif random_aug == 3:
        # CLOCK ROTAION
        aug_data = torch.rot90(concat_data, 3, [3, 4])
        aug_mask = torch.rot90(concat_seg, 3, [3, 4])
    

        if DEBUG==True:
            visualize_clips(concat_seg[0], 0, 'mask')
            aug_visualize_clips(aug_mask[0], 0, 'clock_mask')

    assert aug_data.shape == concat_data.shape
    assert aug_mask.shape == concat_seg.shape


    # ORIGINAL CLIP OUTPUT
    output, predicted_action, feat  = model(concat_data, concat_action)
    # AUGMENTED CLIP OUTPUT
    aug_op, aug_pred_action, aug_feat = model(aug_data, concat_action)
    # print(output.shape, predicted_action.shape, feat.shape)
    # print(aug_op.shape, aug_pred_action.shape, aug_feat.shape)
    # aug_visualize_clips(aug_op[0], 0, 'pred_aug_mask')
    # print("DONE....")

    # SEG LOSS SUPERVISED - ORIGINAL
    labeled_op = output[labeled_vid_index]
    labeled_seg_data = concat_seg[labeled_vid_index]
    sup_seg_loss_1 = criterion1(labeled_op, labeled_seg_data.float().cuda())

    # SEG LOSS SUPERVISED - AUGMENTED
    # labeled_aug_op = aug_op[labeled_vid_index]
    # labeled_aug_seg_data = aug_mask[labeled_vid_index]
    # sup_seg_loss_2 = criterion1(labeled_aug_op, labeled_aug_seg_data.float().cuda())
    
    # CLS LOSS SUPERVISED - ORIGINAL
    labeled_cls = concat_action[labeled_vid_index]
    labeled_pred_action = predicted_action[labeled_vid_index]
    class_loss_1, abs_class_loss_1 = criterion5(labeled_pred_action, labeled_cls, r)

    # CLS LOSS SUPERVISED - AUGMENTED
    # labeled_aug_pred_action = aug_pred_action[labeled_vid_index]
    # class_loss_2, abs_class_loss_2 = criterion5(labeled_aug_pred_action, labeled_cls, r)

    if const_loss == 'jsd':
        consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
        feat += 1e-7
        aug_feat += 1e-7
        cons_loss_a = consistency_criterion(feat.log(), aug_feat.detach()).sum(-1).mean()
        cons_loss_b = consistency_criterion(aug_feat.log(), feat.detach()).sum(-1).mean()
        total_cons_loss = cons_loss_a + cons_loss_b

    elif const_loss == 'l2':
        consistency_criterion = nn.MSELoss()
        total_cons_loss = consistency_criterion(feat, aug_feat)

    elif const_loss == 'l1':
        consistency_criterion = nn.L1Loss()
        total_cons_loss = consistency_criterion(feat, aug_feat)
    
    # seg_loss = sup_seg_loss_1 + sup_seg_loss_2
    seg_loss = sup_seg_loss_1

    # class_loss = class_loss_1 + class_loss_2
    class_loss = class_loss_1
    
    total_loss = wt_seg * seg_loss + wt_cls * class_loss + wt_cons_cls * total_cons_loss

    return (output, predicted_action, concat_seg, concat_action, total_loss, seg_loss, class_loss, total_cons_loss)




def train(model, labeled_train_loader, unlabeled_train_loader, optimizer, scheduler, epoch, r, save_path, writer, wt_seg, wt_cls, wt_cons_cls, const_loss, short=False):
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
    
    start_time = time.time()
    for batch_id, (label_minibatch, unlabel_minibatch)  in enumerate(zip(cycle(labeled_train_loader), unlabeled_train_loader)):
        if short:
            print("this condition")
            if batch_id > 40:
                break
    
        optimizer.zero_grad()
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cc_loss = train_model_interface(label_minibatch, unlabel_minibatch, wt_seg, wt_cls, wt_cons_cls, 
                                                                                                                const_loss, r)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        # print(len(total_loss))
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        class_consistency_loss.append(cc_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        report_interval = 10
        if (batch_id + 1) % report_interval == 0:
            r_total = np.array(total_loss).mean()
            # print(r_total)
            r_seg = np.array(seg_loss).mean()
            r_class = np.array(class_loss).mean()
            r_cc_class = np.array(class_consistency_loss).mean()
            r_acc = np.array(accuracy).mean()
            print('%d/%d  %d/%d  %.3f  %.3f %.3f %.3f  %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class, r_cc_class, r_acc))

            # scheduler.step(r_total)
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
    # r_total = np.array(total_loss).mean()
    
    #file = 'weights' + str(epoch)
    torch.save(model.state_dict(), save_path+".pth")
    print('saved weights to ', save_path+".pth")
    train_total_loss = np.array(total_loss).mean()
    # print(train_total_loss)
    # exit()
    return r, train_total_loss


def validate(model, val_data_loader, epoch, short=False):
    steps = len(val_data_loader)
    # print('validation: batch size ', VAL_BATCH_SIZE, ' ', N_EPOCHS, 'epochs', steps, ' steps ')
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
    parser.add_argument('--const_loss', type=str, help='consistency loss type')
    parser.add_argument('--wt_seg', type=float, default=1, help='segmentation loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='Classification loss weight')
    parser.add_argument('--wt_cons_cls', type=float, default=1, help='class consistency loss weight')
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
    labeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=4, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)

    unlabeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=12, file_id=args.pkl_file_unlabel, percent=percent, use_random_start_frame=False)

    validationset = UCF101DataLoader('validation',[224, 224], VAL_BATCH_SIZE, file_id="test_annots.pkl", use_random_start_frame=False)
    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))
    # print(labeled_trainset[0]['data'].shape)
    # index=0
    # visualize_clips(labeled_trainset[index]['data'], index)

    # exit()
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
    model = CapsNet(pretrained_load=True)
    
    if USE_CUDA:
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1, verbose=True)


    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/activity_detect/caps_net/exp_4_data_aug/train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    
    prev_best_val_loss = 10000
    prev_best_train_loss = 10000
    prev_best_val_loss_model_path = None
    prev_best_train_loss_model_path = None
    r = 0
    for e in tqdm(range(1, N_EPOCHS + 1)):
    
        r, train_loss = train(model, labeled_train_data_loader, unlabeled_train_data_loader, optimizer, scheduler, e, r, save_path, writer, args.wt_seg, args.wt_cls, args.wt_cons_cls, args.const_loss, short=False)

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