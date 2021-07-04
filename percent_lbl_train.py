import sys
import os
import utils
import torch
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from models.capsules_ucf101 import CapsNet

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

from datasets.ucf_dataloader import UCF101DataLoader
from train_utils import SpreadLoss, get_accuracy, DiceLoss, IoULoss


'''
python percent_lbl_train.py --bs 8 --exp_id supervised_20_per_bce_dice>supervised_20_per_bce_dice.out
python percent_lbl_train.py --bs 8 --epochs 50 --exp_id sup_10_bd>supervised_20_per_bce_iou.out

'''

def model_interface(minibatch, r=0):
    r = 0
    data = minibatch['data'].type(torch.cuda.FloatTensor)
    action = minibatch['action'].cuda()
    segmentation = minibatch['segmentation']

    output, predicted_action, feat = model(data, action)
    class_loss, abs_class_loss = criterion_cls(predicted_action, action, r)
    seg_loss1 = criterion_seg_1(output, segmentation.float().cuda())
    seg_loss2 = criterion_seg_2(output, segmentation.float().cuda())
    
    seg_loss = seg_loss1 + seg_loss2
    total_loss =  seg_loss + class_loss
    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss)


def train(model, train_loader, optimizer, epoch, r, save_path, writer, short=False):
    start_time = time.time()
    steps = len(train_loader)
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_loss_sent = []
    accuracy_sent = []
    print('epoch  step    loss   seg    class  accuracy')
    
    start_time = time.time()
    for batch_id, minibatch in enumerate(train_loader):
        if short:
            if batch_id > 40:
                break

        optimizer.zero_grad()
        if (batch_id + 1) % 100 == 0:
            r = (1. * batch_id + (epoch - 1) * steps) / (30 * steps)
    
        output, predicted_action, segmentation, action, loss, s_loss, c_loss = model_interface(minibatch, r)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        report_interval = 10
        if (batch_id + 1) % report_interval == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(seg_loss).mean()
            r_class = np.array(class_loss).mean()
            r_acc = np.array(accuracy).mean()

            print('%d/%d  %d/%d  %.3f  %.3f  %.3f %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class,r_acc))

            # summary writing
            total_step = (epoch-1)*len(train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_seg': r_seg,
                'loss_cls': r_class,
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
    train_total_loss = np.array(total_loss).mean()
    #file = 'weights' + str(epoch)
    torch.save(model.state_dict(), save_path+".pth")
    print('saved weights to ', save_path+".pth")
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
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss = model_interface(minibatch, r)
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
    print('Validation %d  %.3f  %.3f  %.3f %.3f IOU %.3f' % (epoch, r_total, r_seg, r_class, r_acc, average_IOU))
    sys.stdout.flush()
    return r_total



def parse_args():
    parser = argparse.ArgumentParser(description='add_losses')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--bs', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seg_loss', type=str, default='dice', help='dice or iou loss')
    # parser.add_argument('--log', type=str, default='log_prp', help='log directory')
    parser.add_argument('--exp_id', type=str, default='loss_checks', help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, default="train_annots_10_labeled_random.pkl", help='experiment name')
    # parser.add_argument('--pkl_file_unlabel', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)
    
    USE_CUDA = True if torch.cuda.is_available() else False
    TRAIN_BATCH_SIZE = args.bs
    VAL_BATCH_SIZE = args.bs
    N_EPOCHS = args.epochs
    LR = args.lr
    seg_loss_criteria = args.seg_loss
    
    percent = str(100)
    labeled_trainset = UCF101DataLoader('train', [224, 224], batch_size=8, file_id=args.pkl_file_label, percent=percent, use_random_start_frame=False)
    validationset = UCF101DataLoader('validation',[224, 224], VAL_BATCH_SIZE, file_id="test_annots.pkl", use_random_start_frame=False)
    print(len(labeled_trainset), len(validationset))
    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=8,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=8,
        shuffle=False
    )

    print(len(labeled_train_data_loader), len(val_data_loader))
    
    # Load pretrained weights
    model = CapsNet(pretrained_load=True)
    
    if USE_CUDA:
        model = model.cuda()

    # losses
    global criterion_cls
    global criterion_seg_1
    global criterion_seg_2
    criterion_cls = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion_seg_1 = nn.BCEWithLogitsLoss(size_average=True)

    if seg_loss_criteria == 'dice':
        criterion_seg_2 = DiceLoss()

    if seg_loss_criteria == 'iou':
        criterion_seg_2 = IoULoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1, verbose=True)
    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/activity_detect/caps_net/exp_4_data_aug/sup_train_wts', exp_id)
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
        
        r, train_loss = train(model, labeled_train_data_loader, optimizer, e, r, save_path, writer, short=False)

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
