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
from scipy.stats import norm

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



class JHMDB(Dataset):
    def __init__(self, name, clip_shape, file_id, use_random_start_frame=False):

      self._dataset_dir = '/home/dataset/JHMDB'
      self._mask_dir = '/home/dataset/puppet_mask'
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
      
      # self._percent = percent
      self._use_random_start_frame = use_random_start_frame
      self._height = clip_shape[0]
      self._width = clip_shape[1]
      #self._channels = channels
      # self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
      print("Init load_jhmdb_pytorch_multi")
            

    def get_det_annots_prepared(self, file_id):
        
        # training_annot_file = 'trainlist_JHMDB.txt'
        # training_annot_file = 'trainlist.txt'
        training_annot_file =  '../jhmdb_seed_37/' + file_id

        with open(training_annot_file,"r") as rid:
            train_list = rid.readlines()
        for i in range(len(train_list)):
            train_list[i] = train_list[i].rstrip()
            
        print("Training samples from :", training_annot_file)
        
        return train_list
        
        
    def get_det_annots_test_prepared(self):
        # test_annot_file = 'testlist_JHMDB.txt'
        test_annot_file = '../testlist.txt'

        with open(test_annot_file,"r") as rid:
            test_file_list = rid.readlines()
        
        for i in range(len(test_file_list)):
            test_file_list[i] = test_file_list[i].rstrip()
            
        print("Testing samples from :", test_annot_file)
        
        return test_file_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        mask_cls = np.zeros((depth, self._height, self._width, 1))
        
        v_name = self.vid_files[index]
        
        clip, bbox_clip, label, annot_frames = self.load_video(v_name)
        
        bbox_clip = np.reshape(bbox_clip, (bbox_clip.shape[0], bbox_clip.shape[1], bbox_clip.shape[2], 1))

        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
            mask_cls = torch.from_numpy(mask_cls)
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls, 'aug_data': video_rgb}
            return sample
        
        
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
                mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
                mask_cls = torch.from_numpy(mask_cls)
                sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls, 'aug_data': video_rgb}
                return sample
            annot_idx = np.random.randint(0,len(annot_frames))
            selected_annot_frame = annot_frames[annot_idx]
        
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
            mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
            mask_cls = torch.from_numpy(mask_cls)
            sample = {'data':video_rgb,'loc_msk':label_cls,'action':torch.Tensor([0]),'mask_cls':mask_cls, 'aug_data': video_rgb}
            return sample
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)
        
        
        span = (np.arange(depth)*vskip)
        span += start_frame
        #print("Span ", span, ", vs:", clip.shape, ", cs:", bbox_clip.shape)
        video = clip[span]
        bbox_clip = bbox_clip[span]
        closest_fidx = np.argmin(np.abs(span-selected_annot_frame))
        
        sigma = 4/3. # 2.5
        gaus_vals = norm.pdf(np.arange(-4,5), 0, sigma)
        gaus_vals = gaus_vals / np.max(gaus_vals)
        gaus_mid = 5
        if self.name == 'train':
            #start_pos = np.random.randint(0,16)
            start_pos_h = np.random.randint(0,clip_h - 224) #self._height)
            start_pos_w = np.random.randint(0,clip_w - 224) #self._width)
        else:
            # center crop for validation
            start_pos_h = int((clip_h - 224) / 2)
            start_pos_w = int((clip_w - 224) / 2)

        
        final_gaus_mask = np.zeros((depth))
        
        for j in range(video.shape[0]):
            img = video[j]
            img = img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            img = cv2.resize(img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            img = img / 255.
            video_rgb[j] = img
            
            valid_frame = False
            if vskip == 2:
                if span[j] in annot_frames or span[j]+1 in annot_frames:
                    valid_frame = True
            elif vskip == 1:
                if span[j] in annot_frames:
                    valid_frame = True
                    
            if valid_frame:
                bbox_img = bbox_clip[j]
                bbox_img[bbox_img>0] = 255
                bbox_img = bbox_img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224, 0]
                bbox_img = cv2.resize(bbox_img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
                label_cls[j, bbox_img > 0, 0] = 1.
                mask_cls[j, :, :, :] = 1.
                

        horizontal_flipped_video = video_rgb[:, :, ::-1, :]
        
        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)
        
        mask_cls = np.transpose(mask_cls, [3, 0, 1, 2])
        mask_cls = torch.from_numpy(mask_cls)
        
        horizontal_flipped_video = np.transpose(horizontal_flipped_video, [3, 0, 1, 2])
        horizontal_flipped_video = torch.from_numpy(horizontal_flipped_video.copy())

        action_tensor = torch.Tensor([label])        
        
        sample = {'data':video_rgb,'loc_msk':label_cls,'action':action_tensor, 'mask_cls': mask_cls, "aug_data":horizontal_flipped_video}
        return sample


    def load_video(self, video_name):
        video_dir = os.path.join(self._dataset_dir, '%s.avi' % video_name)

        try:
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

            video_reshape = np.zeros((video.shape[0], 256, 256, 3))
            # print(video_reshape.shape)
            for xxx in range(video_reshape.shape[0]):
                video_reshape[xxx] = cv2.resize(video[xxx], (256, 256), interpolation=cv2.INTER_AREA)     
            
                
            if self.name == 'train':
                # For 100%
                mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name))
                # print(mask_dir)
                mat_data = loadmat(mask_dir)
                mask_m = mat_data['part_mask']
                # print(mask_m.shape)

                mask = np.zeros((mask_m.shape[2], 256, 256))
                # print(mask.shape)
                # mask = mask_m
                for m in range(mask_m.shape[2]):
                    mask[m] = cv2.resize(mask_m[:,:,m], (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, -1)
                # print(mask_m.shape)
                annot_frames = np.arange(mask.shape[0])

                
            elif self.name == 'test':            
                # mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name,self._percent))
                mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name))  # for 100%

                mat_data = loadmat(mask_dir)
                mask_m = mat_data['part_mask']
                mask = np.zeros((mask_m.shape[2], 256, 256))
                for m in range(mask_m.shape[2]):
                    mask[m] = cv2.resize(mask_m[:,:,m], (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_m = np.expand_dims(mask, -1)                
                
                #annot_frames = mat_data['annot_frames']
                annot_frames = np.arange(mask.shape[0]) # For 100%
                if len(annot_frames.shape) > 1:
                    annot_frames = annot_frames.reshape(-1)

            else:
                print("Invalid mode ", self.name)
                exit(0)
            
            
        except:
            # video = vread(str(video_dir), num_frames=40)
            print('Error:', str(video_dir))
            print('Error:', str(mask_dir))
            return None, None, None, None
            # print(video.shape)

        label_name = video_name.split('/')[0]
        label = self._class_list.index(label_name)
        
        if self.name == 'train':
            return video_reshape, mask, label, annot_frames
        else:
            return video_reshape, mask_m, label, annot_frames

if __name__ == '__main__':
    import imageio    
    name='train'
    clip_shape=[224,224]
    channels=3
    # batch_size = 5
    dataloader = JHMDB(name, clip_shape, "jhmdb_classes_list_per_30_labeled.txt")
    print(len(dataloader))
    #exit()
    index = 0
    while True:
        sample = dataloader.__getitem__(index)
        
        clip = sample['data']
        bbox_cls = sample['loc_msk']
        bbox_msk = sample['mask_cls']
        # print(clip.shape, bbox_cls.shape, bbox_msk.shape)
        # exit()
        clip, bbox_cls, bbox_msk = clip.numpy(), bbox_cls.numpy(), bbox_msk.numpy()
        clip = np.transpose(clip, [1, 2, 3, 0])
        # print(clip.shape)
        # exit()
        bbox_cls = np.transpose(bbox_cls, [1, 2, 3, 0])
        bbox_msk = np.transpose(bbox_msk, [1, 2, 3, 0])

        if index==7:
            with imageio.get_writer('./results/orig_{:02d}_gt.gif'.format(index), mode='I') as writer:
                for i in range(clip.shape[0]):
                    image = (clip[i]*255).astype(np.uint8)
                    image = image[...,::-1].copy()
                    writer.append_data(image) 
                    
            # with imageio.get_writer('./results/orig_cls_{:02d}_gt.gif'.format(index), mode='I') as writer:
            #     for i in range(clip.shape[0]):
            #         image = (clip[i,:,:,0]*255).astype(np.uint8)
            #         cl_mask = (bbox_cls[i,:,:,0]*255).astype(np.uint8)
            #         # cl_mask[cl_mask>0] = 255
            #         # print(image.shape,clip_mask[i,:,:,0].shape)
            #         # image = cv2.drawContours(image, clip_mask[i,:,:,0][0], -1, (0 , 255, 0), 3)
            #         image = cv2.bitwise_and(image, image, mask=cl_mask)
            #         writer.append_data(image) 

            # with imageio.get_writer('./results/orig_mask_{:02d}_gt.gif'.format(index), mode='I') as writer:
            #     for i in range(clip.shape[0]):
            #         image = (clip[i,:,:,0]*255).astype(np.uint8)
            #         cl_mask = (bbox_msk[i,:,:,0]*255).astype(np.uint8)
            #         # print(image.shape,clip_mask[i,:,:,0].shape)
            #         # image = cv2.drawContours(image, clip_mask[i,:,:,0][0], -1, (0 , 255, 0), 3)
            #         image = cv2.bitwise_and(image, image, mask=cl_mask)
            #         writer.append_data(image) 

            exit()
        
        index += 1         
        # exit() 
    
