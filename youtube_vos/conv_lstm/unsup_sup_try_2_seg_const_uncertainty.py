import config
import torch
import os
import random
import time
import pprint
import argparse

from torch.autograd.variable import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import VOSModel
from PIL import Image
from pathlib import Path
#from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from itertools import cycle
from datasets.dataloader import TrainDataset, ValidationDataset


def print_options(options):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint("All the options are as follows:")
    pp.pprint(options)
    print('\n', flush=True) #To flush the output stream of pretty print - pp doesn't have flush option
    return

def get_accuracy(y_pred, y):
    y_argmax = torch.round(y_pred)

    return torch.mean((y_argmax==y).type(torch.float))


def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).mean()


def save_images(y_pred, y):
    y = y * 255

    y_argmax = torch.round(y_pred)
    y_argmax = y_argmax * 255

    prints = y.cpu().detach().numpy().astype(np.uint8)
    #train = TrainDataset()
    #palette = Image.open(train.train_anns[0][0][0])
    for i in range(len(prints[0][0])):
        try:
            c = Image.fromarray(prints[0][0][i], mode='P')#.resize(size=palette.size)
            #c.putpalette(palette.getpalette())
            c.save('./save_images/%d_%d.png' % (i, config.epoch), "PNG", mode='P')

        except:
            print('error saving %d_%d.png' % (i, config.epoch))
    prints2 = y_argmax.cpu().detach().numpy().astype(np.uint8)
    for i in range(len(prints2[0][0])):
        try:
            c = Image.fromarray(prints2[0][0][i], mode='P')#.resize(size=palette.size)
            #c.putpalette(palette.getpalette())
            c.save('./save_images/%d_%d pred.png' % (i, config.epoch), "PNG", mode='P')
        except:
            print('error saving %d_%d pred.png' % (i, config.epoch))


def measure_pixelwise_uncertainty(pred):
    count = 0
    batch_variance = np.zeros_like(pred.cpu().detach().numpy(), dtype=np.float64)

    for zz in range(0, pred.shape[0]):
        m_temp = pred[zz][0]
        clip_variance = np.zeros_like(batch_variance[0][0])

        # RANGE VARIES FROM 0 TO N_FRAMES
        for temp_cnt in range(clip_variance.shape[0]):
            # FIRST FRAME
            if temp_cnt-1<0:
                temp_var = m_temp[temp_cnt:temp_cnt+2]
            # LAST FRAME
            elif temp_cnt+1>(clip_variance.shape[0] - 1):
                temp_var = m_temp[temp_cnt-1:]
            # FRAMES IN BETWEEN
            else:
                temp_var = m_temp[temp_cnt-1:temp_cnt+2]

            temp_var = np.var(temp_var.cpu().detach().numpy(), axis=0)
            temp_var -= temp_var.min()
            temp_var /= (temp_var.max() - temp_var.min())

            clip_variance[temp_cnt] = temp_var

        clip_variance = np.reshape(clip_variance, (1, clip_variance.shape[0], clip_variance.shape[1], clip_variance.shape[2]))
        batch_variance[zz] = clip_variance
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance

            
def train(args, model, epoch, labeled_dloader, unlabeled_dloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    losses, accs = [], []

    start_time = time.time()


    for i, (labeled_sample, unlabeled_sample) in enumerate(zip(cycle(labeled_dloader), unlabeled_dloader)):
        lbl_inputs_batch, lbl_anns_batch, lbl_vid_anns_indeces_batch, lbl_anns_mask_batch = labeled_sample
        lbl_vid_anns_indeces_batch = lbl_vid_anns_indeces_batch.cpu().detach().numpy()

        unlbl_inputs_batch, unlbl_anns_batch, unlbl_vid_anns_indeces_batch, unlbl_anns_mask_batch = unlabeled_sample
        unlbl_vid_anns_indeces_batch = unlbl_vid_anns_indeces_batch.cpu().detach().numpy()

        optimizer.zero_grad()
        lbl_inputs_batch, lbl_anns_batch = lbl_inputs_batch.type(torch.float), lbl_anns_batch.type(torch.float)
        lbl_anns_mask_batch = lbl_anns_mask_batch.type(torch.float)

        unlbl_inputs_batch, unlbl_anns_batch = unlbl_inputs_batch.type(torch.float), unlbl_anns_batch.type(torch.float)
        unlbl_anns_mask_batch = unlbl_anns_mask_batch.type(torch.float)

        # print(lbl_inputs_batch.shape) #(BS, 3, 32, 224, 224)
        # print(lbl_inputs_batch[:, :, 0].shape) #(BS, 3, 224, 224)
        # print(lbl_anns_batch.shape) #(BS, 1, 32, 224, 224)
        # print(lbl_anns_mask_batch)  # (BS, 1, 32, 1, 1)
        # print(lbl_vid_anns_indeces_batch.shape) #(BS, 1, 7)
        # print(unlbl_inputs_batch.shape) #(BS, 3, 32, 224, 224)
        # print(unlbl_inputs_batch[:, :, 0].shape) #(BS, 3, 224, 224)
        # print(unlbl_anns_batch.shape) #(BS, 1, 32, 224, 224)
        # print(unlbl_vid_anns_indeces_batch.shape) #(BS, 1, 7)
        # exit()
   
        lbl_inputs_batch = lbl_inputs_batch.cuda()
        lbl_anns_batch = lbl_anns_batch.cuda()
        lbl_anns_mask_batch = lbl_anns_mask_batch.cuda()

        unlbl_inputs_batch = unlbl_inputs_batch.cuda()
        unlbl_anns_batch = unlbl_anns_batch.cuda()
        unlbl_anns_mask_batch = unlbl_anns_mask_batch.cuda()

        concat_inputs_batch = torch.cat([lbl_inputs_batch, unlbl_inputs_batch], axis=0)
        concat_anns_batch = torch.cat([lbl_anns_batch, unlbl_anns_batch], axis=0)
        concat_anns_mask_batch = torch.cat([lbl_anns_mask_batch, unlbl_anns_mask_batch], axis=0)
        
        ones_tensor = torch.ones(len(lbl_inputs_batch))
        zeros_tensor = torch.zeros(len(unlbl_inputs_batch))
        concat_labels = torch.cat([ones_tensor, zeros_tensor], dim=0).cuda()
        random_indices = torch.randperm(len(concat_labels))

        concat_inputs_batch = concat_inputs_batch[random_indices, :, :, :, :]
        concat_anns_batch = concat_anns_batch[random_indices, :, :, :, :]
        concat_anns_mask_batch = concat_anns_mask_batch[random_indices, :, :, :, :]
        concat_labels = concat_labels[random_indices]

        # print(concat_inputs_batch.shape)
        concat_aug_inputs_batch = torch.flip(concat_inputs_batch, [4])
        concat_aug_anns_batch = torch.flip(concat_anns_batch, [4])

        #####################################################
        # Need to check concat_anns_mask_batch
        #####################################################

        labeled_vid_index = torch.where(concat_labels==1)[0]

        y_pred_logits, y_pred, _ = model(concat_inputs_batch, concat_inputs_batch[:, :, 0], concat_anns_batch[:, :, 0])
        y_pred_logits_fl, y_pred_fl, _ = model(concat_aug_inputs_batch, concat_aug_inputs_batch[:, :, 0], concat_aug_anns_batch[:, :, 0])
        y_pred_logits_fl = torch.flip(y_pred_logits_fl, [4])

        # SEG LOSS SUPERVISED
        labeled_pred = y_pred_logits[labeled_vid_index]
        labeled_gt = concat_anns_batch[labeled_vid_index]
        labeled_gt_anns_mask = concat_anns_mask_batch[labeled_vid_index]

        # print(labeled_pred.shape, labeled_gt.shape, labeled_gt_anns_mask.shape)

        #print('Finished prediction %d...' % i)
        seg_loss = criterion(labeled_pred, labeled_gt) * labeled_gt_anns_mask
        seg_loss = seg_loss.sum() / (224 * 224 * (args.batch_size//2))

        # cons_loss = consistency_criterion(y_pred_logits, y_pred_logits_fl)

        batch_variance_orig_clip = measure_pixelwise_uncertainty(y_pred_logits)
        batch_variance_orig_clip = batch_variance_orig_clip.type(torch.cuda.FloatTensor)
        loss_wt = weighted_mse_loss(y_pred_logits_fl, y_pred_logits, batch_variance_orig_clip)

        cons_loss = loss_wt
        
        loss = seg_loss + args.wt_cons * cons_loss
        

        if seg_loss <= 0:
            print(seg_loss, y_pred.max(), y_pred.min(), y_pred.mean())
            print(concat_anns_mask_batch.max(), concat_anns_mask_batch.min(), concat_anns_mask_batch.mean())
            exit()
        # acc = get_accuracy(y_pred[:, :, vid_anns_indeces_batch[0][0], :, :], anns_batch[:, :, vid_anns_indeces_batch[0][0], :, :])

        # acc = get_accuracy(y_pred[:, :, lbl_vid_anns_indeces_batch[0][0], :, :], lbl_anns_batch[:, :, lbl_vid_anns_indeces_batch[0][0], :, :])
        acc = get_accuracy(labeled_pred[:, :, lbl_vid_anns_indeces_batch[0][0], :, :], lbl_anns_batch[:, :, lbl_vid_anns_indeces_batch[0][0], :, :])

        # if i == 0:
        #     # save_images(y_pred[:, :, vid_anns_indeces_batch[0][0], :, :], anns_batch[:, :, vid_anns_indeces_batch[0][0], :, :])
        #     save_images(y_pred[:, :, unlbl_vid_anns_indeces_batch[0][0], :, :], unlbl_anns_batch[:, :, unlbl_vid_anns_indeces_batch[0][0], :, :])
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()
		
        losses.append(loss.item())
        accs.append(acc.item())
        
        # if i % 40 == 0:
        if i % args.pf == 0:

            # print(i,"/",len(unlabeled_dloader))

            avg_loss = running_loss / args.pf
            # avg_acc = running_acc / (args.pf*args.batch_size)
            avg_acc = running_acc/ args.pf
            print(f'[TRAIN] epoch-{epoch}, batch-{i:4d}/{len(unlabeled_dloader)}, loss: {avg_loss:.3f}, acc: {avg_acc:.3f}')
            # step = (epoch-1)*len(unlabeled_dloader) + i
            running_loss = 0.0
            running_acc = 0.0

    print('Finished predictions...')
    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    return float(np.mean(losses)), float(np.mean(accs))


def parse_args():
    parser = argparse.ArgumentParser(description='vos_exps')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--n_frames', type=int, default=32, help='frames count')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight decay')
    parser.add_argument('--pretrained', type=bool, default=True, help='loading pretrained model')
    parser.add_argument('--resume', type=bool, default=False, help='loading pretrained model')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seg_loss', type=str, default='dice', help='dice or iou loss')
    # parser.add_argument('--log', type=str, default='log_prp', help='log directory')
    parser.add_argument('--exp_id', type=str, default='debugs', help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, help='experiment name')
    parser.add_argument('--pkl_file_unlabel', type=str, help='experiment name')
    parser.add_argument('--const_loss', type=str, help='consistency loss type')
    parser.add_argument('--wt_seg', type=float, default=1, help='segmentation loss weight')
    parser.add_argument('--wt_cons', type=float, default=1, help='class consistency loss weight')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    # parser.add_argument('--pretrained', type=bool, )
    args = parser.parse_args()
    return args



def run_experiment():
    print("runnning...")

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

    data_root = '/datasets/YouTube-VOS/2019/train'          # Original 100%
    # meta_data_root = '/home/arana/projects/REUVOS'                # Modified percentage data 
    # meta_data_root = '/datasets/YouTube-VOS/2019/train'
    meta_data_root = 'json_files'
    labeled_json_file = 'meta_20percent_labeled.json'
    unlabeled_json_file = 'meta_80percent_unlabeled.json'

    labeled_train_dataset = TrainDataset(root=data_root, meta_root = meta_data_root, json_file = labeled_json_file)
    unlabeled_train_dataset = TrainDataset(root=data_root, meta_root = meta_data_root, json_file = unlabeled_json_file)

    print("dataset loaded...")
    print(len(labeled_train_dataset), len(unlabeled_train_dataset))     # 347, 3124
    
    labeled_dloader = DataLoader(labeled_train_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=8)
    unlabeled_dloader = DataLoader(unlabeled_train_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=8)

    print(len(labeled_dloader), len(unlabeled_dloader))
    # exit()

    # global consistency_criterion

    # consistency_criterion = nn.MSELoss()

    if config.bce_w_logits:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.BCELoss(reduction='none')

    model = VOSModel()

    if USE_CUDA:
        model.cuda()
    
    if args.resume:
        load_model = torch.load(config.resume_model_path)
        model.load_state_dict(load_model['state_dict'])
        print("Loaded weights from ", config.resume_model_path)


    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    print("model loaded...")

    exp_id = args.exp_id
    save_path = os.path.join('/home/akumar/vos/lstm/save_models', exp_id)
    model_save_dir = os.path.join(save_path,time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    best_loss = 1000000

    losses = []
    accuracies = []
    for epoch in range(1, args.epochs + 1):
    # for epoch in range(1, args..n_epochs + 1):

        print('Epoch:', epoch)
        config.epoch = epoch
        time_start = time.time()
        loss, acc = train(args, model, epoch, labeled_dloader, unlabeled_dloader, criterion, optimizer)
        print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
        print('Finished training. Loss: ',  loss, ' Accuracy: ', acc)
        losses.append(loss)
        accuracies.append(acc)
        if epoch % 2 == 0:
            print('Model Improved -- Saving.')
            best_loss = loss

            save_file_path = os.path.join(model_save_dir, 'model_{}_{:.4f}.pth'.format(epoch, loss))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))
	
    save_file_path = os.path.join(model_save_dir, 'model_{}_{:.4f}.pth'.format(epoch, loss))
    states = {
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
	}

    torch.save(states, save_file_path)

    print('Training Finished')
    # multiple line plot
    # multiple line plot
    #plt.plot(losses, label='loss')
    #plt.plot(accuracies, label='accuracy')
    #plt.legend()
    #plt.axis([0, 99, 0, 1])
    #plt.show()


if __name__ == '__main__':
    run_experiment()


'''0
try:
            Path.mkdir(log_path, parents=True, exist_ok=True)
        except OSError as e:
            if e.errno == errno.EEXIST:
                logger.warning(f'Tensorboard log directory already exists: {str(log_path)}')
                for f in log_path.rglob("*"):
                    f.unlink()

            else:
                raise


'''