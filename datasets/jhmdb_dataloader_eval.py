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

Loads in videos for the 21 class subset of JHMDB-21.


'''



class JHMDB(Dataset):
    'Prunes JHMDB data'
    def __init__(self, name, clip_shape, file_id, use_random_start_frame=False):
      self._dataset_dir = '/path/dataset/videos/JHMDB'
      self._mask_dir = '/path/dataset/anns/puppet_mask'
      #self.get_det_annotations()       # To prepare pickle file for annots
      
      self._class_list = ['brush_hair','catch','clap','climb_stairs',
                            'golf','jump','kick_ball','pick','pour',
                            'pullup','push','run','shoot_ball','shoot_bow',
                            'shoot_gun','sit','stand','swing_baseball',
                            'throw','walk','wave']
                            
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared(file_id)
          self.shuffle = True
          self.name = 'train'
      else:
          self.vid_files = self.get_det_annots_test_prepared()
          self.shuffle = False
          self.name = 'test'

      self._use_random_start_frame = use_random_start_frame
      self._height = clip_shape[0]
      self._width = clip_shape[1]
      #self._channels = channels
      # self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, file_id):
        
        training_annot_file =  '../' + file_id

        with open(training_annot_file,"r") as rid:
            train_list = rid.readlines()
        for i in range(len(train_list)):
            train_list[i] = train_list[i].rstrip()
            
        print("Training samples from :", training_annot_file)
        
        return train_list        
        
    def get_det_annots_test_prepared(self):
        test_annot_file = '../testlist.txt'
        with open(test_annot_file,"r") as rid:
            test_file_list = rid.readlines()
        
        for i in range(len(test_file_list)):
            test_file_list[i] = test_file_list[i].rstrip()
            
        # print("Testing samples from :", test_annot_file)
        
        return test_file_list  

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        v_name = self.vid_files[index]
        # print(v_name)
        clip, bbox_clip, label = self.load_video(v_name)
        # print(clip.shape)
        # Center crop
        frames, h, w, _ = clip.shape     
        margin_h = h - self._height
        h_crop_start = int(margin_h/2)
        margin_w = w - self._width
        w_crop_start = int(margin_w/2)
        
        clip = clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :] / 255.
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :]
        
        return clip, bbox_clip, label, v_name


    def load_video(self, video_name):
        video_dir = os.path.join(self._dataset_dir, '%s.avi' % video_name)
        mask_dir = os.path.join(self._mask_dir, '%s/puppet_mask.mat' % video_name)
        
        # print(type(video_dir))
        try:
            # print(str(video_dir))
            #video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
            cap = cv2.VideoCapture(video_dir)
            video = []
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break 
                video.append(frame)
                
            video=np.array(video)

            # print(video.shape)            
            mat_data = loadmat(mask_dir)
            mask = mat_data['part_mask']
        except:
            # video = vread(str(video_dir), num_frames=40)
            print('Error:', str(video_dir))
            return None, None, None
            # print(video.shape)

        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label_name = video_name.split('/')[0]
        label = self._class_list.index(label_name)
        
        mask_m = mat_data['part_mask']
        mask_m = np.transpose(mask_m, [2, 0, 1])
        bbox = np.expand_dims(mask_m, -1)

        return video, bbox, label

