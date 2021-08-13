import config
import torch
import os
import time
import random
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

from datasets.dataloader import TrainDataset, ValidationDataset


def get_accuracy(y_pred, y):
    y_argmax = torch.round(y_pred)

    return torch.mean((y_argmax==y).type(torch.float))

def save_images(y_pred, y):
    y = y * 255

    y_argmax = torch.round(y_pred)
    y_argmax = y_argmax * 255

    prints = y.cpu().detach().numpy().astype(np.uint8)
    #train = TrainDataset()
    #palette = Image.open(train.train_annotations[0][0][0])
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
            
def train(args, model, epoch, dloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    losses, accs = [], []

    for i, sample in enumerate(dloader):
        video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch, video_annotations_mask_batch = sample
        video_annotations_indeces_batch = video_annotations_indeces_batch.cpu().detach().numpy()

        optimizer.zero_grad()
        video_inputs_batch, video_annotations_batch = video_inputs_batch.type(torch.float), video_annotations_batch.type(torch.float)
        video_annotations_mask_batch = video_annotations_mask_batch.type(torch.float)
   
        if config.use_cuda:
            video_inputs_batch = video_inputs_batch.cuda()
            video_annotations_batch = video_annotations_batch.cuda()
            video_annotations_mask_batch = video_annotations_mask_batch.cuda()

        if config.use_fixes:
            n_frames = args.n_frames//2
            clip1 = video_inputs_batch[:, :, :n_frames]
            clip2 = video_inputs_batch[:, :, n_frames:]
            print(clip1.shape)
            print(clip2.shape)

            y_pred_logits1, y_pred1, y_hidden_state1 = model(clip1 , video_inputs_batch[:, :, 0], video_annotations_batch[:, :, 0])

            y_pred_logits2, y_pred2, _ = model(clip2, video_inputs_batch[:, :, 0], video_annotations_batch[:, :, 0], y_hidden_state1)

            y_pred_logits = torch.cat((y_pred_logits1, y_pred_logits2), 2)

            y_pred = torch.cat((y_pred1, y_pred2), 2)
        else:
            y_pred_logits, y_pred, _ = model(video_inputs_batch, video_inputs_batch[:, :, 0], video_annotations_batch[:, :, 0])

        #print('Finished prediction %d...' % i)
        if config.bce_w_logits:
            loss = criterion(y_pred_logits, video_annotations_batch) * video_annotations_mask_batch
        else:
            loss = criterion(y_pred, video_annotations_batch) * video_annotations_mask_batch
        # loss = loss.sum() / (224 * 224 * config.batch_size)
        loss = loss.sum() / (224 * 224 * args.batch_size)
        if loss <= 0:
            print(loss, y_pred.max(), y_pred.min(), y_pred.mean())
            print(video_annotations_mask_batch.max(), video_annotations_mask_batch.min(), video_annotations_mask_batch.mean())
            exit()
        acc = get_accuracy(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :])
        # if i == 0:
        #     save_images(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()
		
        losses.append(loss.item())
        accs.append(acc.item())
        
        # if i % 40 == 0:
        #     print(i,"/",len(dloader))
        if i % args.pf == 0:

            # print(i,"/",len(unlabeled_dloader))

            avg_loss = running_loss / args.pf
            avg_acc = running_acc / (args.pf*args.batch_size)
            print('[TRAIN] epoch-{}, batch-{}/{}, loss: {:.3f}, acc: {:.3f}'.format(epoch, i, len(dloader), avg_loss, avg_acc))
            # step = (epoch-1)*len(unlabeled_dloader) + i
            running_loss = 0.0
            running_acc = 0.0

    print('Finished predictions...')
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
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment name')
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
    meta_data_root = './json_files'
    json_file = 'meta_20percent_labeled.json'
    # json_file = 'meta.json'
    train_dataset = TrainDataset(root=data_root, meta_root = meta_data_root, json_file = json_file)
    print("dataset loaded...")
    print(len(train_dataset))
    
    if config.bce_w_logits:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.BCELoss(reduction='none')
    model = VOSModel()
    
    if args.resume:
        load_model = torch.load(config.resume_model_path)
        model.load_state_dict(load_model['state_dict'])
        print("Loaded weights from ", config.resume_model_path)

    # if config.use_cuda:
    #     model.cuda()

    if USE_CUDA:
        model.cuda()

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
        # dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        dloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

        #video_inputs_batch, video_annotations_batch, video_annotations_indeces_batch = train_dataset.Datalaoder()

        loss, acc = train(args, model, epoch, dloader, criterion, optimizer)
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

            # try:
            #     os.mkdir(config.save_dir)
            # except:
            #     pass

            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))
	
    save_file_path = os.path.join(model_save_dir, 'model_{}_{:.4f}.pth'.format(epoch, loss))
    states = {
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
	}

    # try:
    #     os.mkdir(config.save_dir)
    # except:
    #     pass

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