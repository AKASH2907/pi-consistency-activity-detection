import os
import time
import numpy as np
import random
from threading import Thread
from scipy.io import loadmat
from skvideo.io import vread
import pdb
import torch
from torch.utils.data import Dataset
import pickle
import cv2

'''

Loads in videos for the 24 class subset of UCF-101.

The data is assumed to be organized in a folder (dataset_dir):
-Subfolder UCF101_vids contains the videos
-Subfolder UCF101_Annotations contains the .mat annotation files

UCF101DataLoader loads in the videos and formats their annotations on seperate threads.
-Argument train_or_test determines if you want to load in training or testing data
-Argument sec_to_wait determines the amount of time to wait to load in data
-Argument n_threads determines the number of threads which will be created to load data

Calling .get_video() returns (video, bboxes, label)
-The videos are in the shape (F, H, W, 3).
-The bounding boxes are loaded in as heat maps of size (F, H, W, 1) where 1 is forground and 0 is background.
-The label is an integer corresponding to the action class.

'''



class UCF101DataLoader(Dataset):
    'Prunes UCF101-24 data'
    def __init__(self, name, clip_shape, file_id, use_random_start_frame=False):
        self._dataset_dir = 'DATA_PATH'

        if name == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.name = 'train'

        else:
            self.vid_files = self.get_det_annots_test_prepared(file_id)
            self.shuffle = False
            self.name = 'test'

        self._use_random_start_frame = use_random_start_frame
        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, file_id):
        import pickle
        
        training_annot_file = "../data_subset_pkl_files/"+ file_id
        
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        
        return training_annotations
        
        
    def get_det_annots_test_prepared(self, file_id):
        import pickle    
        file_id = "test_annots.pkl"
        testing_anns  = "../" + file_id
        with open(testing_anns, 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)
            
        return testing_annotations    
    
    


    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))     # 8, 224,224, 3
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        
        v_name, anns= self.vid_files[index]
        
        clip, bbox_clip, label, annot_frames, labeled_vid = self.load_video(v_name, anns)

        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            labeled_vid = 0
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid':labeled_vid}
            return sample

        #vlen = clip.shape[0]
        vlen, clip_h, clip_w, _ = clip.shape
        vskip = 2
        
        if len(annot_frames) == 1:
            selected_annot_frame = annot_frames[0]
        else:
            if len(annot_frames) <= 0:
                print('annot index error for', v_name, ', ', len(annot_frames), ', ', annot_frames)
                video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
                video_rgb = torch.from_numpy(video_rgb)            
                label_cls = np.transpose(label_cls, [3, 0, 1, 2])
                label_cls = torch.from_numpy(label_cls)
                labeled_vid = 0
                sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid':labeled_vid}

                return sample
            annot_idx = np.random.randint(0,len(annot_frames))
            selected_annot_frame = annot_frames[annot_idx]
            # print(f'selected Frame: {selected_annot_frame}')
        start_frame = selected_annot_frame - int((depth * vskip)/2)

        if start_frame < 0:
            vskip = 1
            start_frame = selected_annot_frame - int((depth * vskip)/2)
            if start_frame < 0:
                start_frame = 0
                vskip = 1
        if selected_annot_frame >= vlen:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            labeled_vid = 0
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'label_vid': labeled_vid}
            return sample
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)
        
        # frame index to chose - 0, 2, 4, ..., 16
        span = (np.arange(depth)*vskip)

        # frame_ids
        span += start_frame
        video = clip[span]
        bbox_clip = bbox_clip[span]
        closest_fidx = np.argmin(np.abs(span-selected_annot_frame))
        
        if self.name == 'train':
            # take random crops for training
            start_pos_h = np.random.randint(0,clip_h - 224) #self._height)
            start_pos_w = np.random.randint(0,clip_w - 224) #self._width)
        else:
            # center crop for validation
            start_pos_h = int((clip_h - 224) / 2)
            start_pos_w = int((clip_w - 224) / 2)
        
        for j in range(video.shape[0]):
            img = video[j]
            img = img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            img = cv2.resize(img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            img = img / 255.
            video_rgb[j] = img
            
            bbox_img = bbox_clip[j]
            bbox_img = bbox_img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            bbox_img = cv2.resize(bbox_img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            label_cls[j, bbox_img > 0, 0] = 1.
                       
        
        horizontal_flipped_video = video_rgb[:, :, ::-1, :]
        # horizontal_flipped_label_cls = label_cls[:,:,::-1,:]

        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)

        horizontal_flipped_video = np.transpose(horizontal_flipped_video, [3, 0, 1, 2])
        horizontal_flipped_video = torch.from_numpy(horizontal_flipped_video.copy())
        
        action_tensor = torch.Tensor([label])

        sample = {'data':video_rgb,'loc_msk':label_cls,'action':action_tensor, "aug_data":horizontal_flipped_video, "label_vid": labeled_vid}

        return sample


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        try:
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
            # print(video.shape)
        except:
            print('Error:', str(video_dir))
            return None, None, None, None, None

        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        labeled_vid = -1
        #multi frame mode
        annot_idx = 0
        if len(annotations) > 1:
            annot_idx = np.random.randint(0,len(annotations))
        multi_frame_annot = []      # annotations[annot_idx][4]
        bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        
        for ann in annotations:
            multi_frame_annot.extend(ann[4])
            start_frame, end_frame, label, labeled_vid = ann[0], ann[1], ann[2], ann[5]
            collect_annots = []
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    bbox[f, y:y+h, x:x+w, :] = 1
                    if f in ann[4]:
                        collect_annots.append([x,y,w,h])
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
            # Expect to have collect_annots with same length as annots for this set 
            # [ c, c, c, c, c, c ....]
            select_annots = ann[4]
            select_annots.sort()
            if len(collect_annots) == 0:
                continue
                
            # x_min, y_min, width, height
            [x, y, w, h] = collect_annots[0]
            if len(collect_annots) == 1:
                bbox_annot[start_frame:end_frame, y:y+h, x:x+w, :] = 1
            else:
                bbox_annot[start_frame:select_annots[0], y:y+h, x:x+w, :] = 1
                for i in range(len(collect_annots)-1):
                    frame_diff = select_annots[i+1] - select_annots[i]
                    if frame_diff > 1:
                        [x, y, w, h] = collect_annots[i]
                        pt1 = np.array([x, y, x+w, y+h])
                        [x, y, w, h] = collect_annots[i+1]
                        pt2 = np.array([x, y, x+w, y+h])
                        points = np.linspace(pt1, pt2, frame_diff).astype(np.int32)
                        for j in range(points.shape[0]):
                            [x1, y1, x2, y2] = points[j]
                            bbox_annot[select_annots[i]+j, y1:y2, x1:x2, :] = 1
                    else:
                        [x, y, w, h] = collect_annots[i]
                        bbox_annot[select_annots[i], y:y+h, x:x+w, :] = 1
                [x, y, w, h] = collect_annots[-1]
                bbox_annot[select_annots[-1]:end_frame, y:y+h, x:x+w, :] = 1
            
        multi_frame_annot = list(set(multi_frame_annot))
        if self.name == 'train':
            return video, bbox, label, multi_frame_annot, labeled_vid
        else:
            return video, bbox, label, multi_frame_annot, labeled_vid


