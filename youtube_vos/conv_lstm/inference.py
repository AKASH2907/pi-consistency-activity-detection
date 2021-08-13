import config
import torch
import os
import random
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from model import VOSModel
from datasets.dataloader import ValidationDataset
from torch.utils.data import DataLoader

from pathlib import Path
# def inference(model_path = './SavedModels/folder/model.pth'):
def inference():


    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='experiment name')
    parser.add_argument('--exp_id', type=str, required=True, help='save models name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    criterion = nn.BCELoss(reduction='mean')
    model = VOSModel()
    load_model = torch.load(args.model_path)
    model.load_state_dict(load_model['state_dict'])
    
    # if config.use_cuda:
    model.cuda()
    model.eval()
    print("model loaded...")
    
    clip_skip_rate = 15 # 30
    with torch.no_grad():
        valid_dataset = ValidationDataset(root='/datasets/YouTube-VOS/2019/valid')
        # valid_dataset = ValidationDataset(root='/datasets/YouTube-VOS/2019/test')
        dloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
        print('dataset loaded...')
        for i, sample in enumerate(dloader):
            video_inputs_batch, video_annotations_batch, video_indeces_batch = sample
            video_indeces_batch = video_indeces_batch.cpu().detach().numpy()
            # if config.use_cuda:
            video_inputs_batch = video_inputs_batch.cuda()
            video_annotations_batch = video_annotations_batch.cuda()

            segs_concat = np.array([])

            for objects in range(len(valid_dataset.valid_frames[i])):
                start_frame = 0
                num_frames = 32
                y_pred_concat = np.array([])
                while start_frame < (len(valid_dataset.valid_frames[i][objects]) - 1):
                    if(start_frame == 0):
                        start_frame = int(video_indeces_batch[0][0][objects])
                        last_frame = video_annotations_batch[:, :, objects]
                        if start_frame > len(y_pred_concat):
                            y_pred_concat = torch.from_numpy(np.zeros((1, 1, start_frame, 224, 224)))
                    else:
                        last_frame = y_pred[:, :, clip_skip_rate]
                    if (start_frame + num_frames) >= len(valid_dataset.valid_frames[i][objects]):
                        num_frames = len(valid_dataset.valid_frames[i][objects]) - start_frame
                    frame_selection = list(range(start_frame, start_frame+num_frames))
                    
                    while len(frame_selection) < 32:
                        frame_selection.append(frame_selection[-1])
                    if start_frame == 0 or not config.use_hidden_state:
                        _, y_pred, hidden_state = model(video_inputs_batch[:, :, frame_selection, :, :], video_inputs_batch[:, :, start_frame], last_frame)
                    elif config.use_hidden_state:
                        _, y_pred, hidden_state = model(video_inputs_batch[:, :, frame_selection, :, :], video_inputs_batch[:, :, start_frame], last_frame, hidden_state)
                    #loss = criterion(y_pred[:, :, video_annotations_indeces_batch[0][0], :, :], video_annotations_batch[:, :, video_annotations_indeces_batch[0][0], :, :])
                    #print(loss.item())
                    # y_pred shape [1, 1, 32, 224, 224]
                    if y_pred_concat.size == 0:
                        y_pred_concat = torch.round(y_pred).cpu().numpy()
                    else:
                        y_pred_np = torch.round(y_pred).cpu().numpy()
                        for index in range(start_frame, len(y_pred_concat[0][0])):
                            y_pred_concat[0][0][index] = y_pred_np[0][0][0]
                            y_pred_np = np.delete(y_pred_np, 0, axis=2)
                        y_pred_concat = np.concatenate((y_pred_concat, y_pred_np), axis=2)
                    start_frame += clip_skip_rate
                frames_to_use = list(range(len(valid_dataset.valid_frames[i][0])))
                y_pred_concat = y_pred_concat[:, :, frames_to_use, :, :]
                if segs_concat.size == 0:
                    test_ann = (np.ones((y_pred_concat.shape)) *.000009)
                    segs_concat = np.concatenate((test_ann, y_pred_concat), axis=1)
                else:
                    segs_concat = np.concatenate((segs_concat, y_pred_concat), axis=1)

            segs_concat = (segs_concat.squeeze(0))
            mask_for_frames = np.argmax(segs_concat, axis=0).astype(dtype=np.uint8)
            print("finished objects segmentation...")
            video_file = valid_dataset.valid_frames[i][0][0].split('/')


            # CREATE EXP DIRECTORY
            dirs = args.exp_id + '/Annotations/%s/' % video_file[-2]
            Path(dirs).mkdir(parents=True, exist_ok=True)

            palette = Image.open(valid_dataset.valid_annotations[i][0][0])
            
            for images in range(len(valid_dataset.valid_frames[i][0])):
                img_file = valid_dataset.valid_frames[i][0][images].split('/')
                c = Image.fromarray(mask_for_frames[images], mode='P').resize(size=palette.size)
                c.putpalette(palette.getpalette())
                img_path = dirs + img_file[-1]
                img_path = img_path.replace('jpg', 'png')
                c.save(img_path, "PNG", mode='P')
            print("segmentation saved...")
    
if __name__ == '__main__':
    
    # inference(config.model_path)
    inference()

    '''
    frame_folder = 'frames'
    seg_folder = 'segs'
    frames = os.listdir(os.path.join('./Vids/', frame_folder))
    segs = os.listdir(os.path.join('./Vids/', seg_folder))
    frames.sort()
    segs.sort()
    for frame in range(len(frames)):
        image = Image.open('./Vids/frames/'+frames[frame])
        mask = Image.open('./Vids/segs/' + segs[frame])
        mask = mask.convert('RGBA')
        transparent = mask.convert("L")
        final = Image.composite(mask, image, transparent)
        final.save('./Vids/%d_%d.png' % (1, frame), "PNG", mode='P')
    '''
