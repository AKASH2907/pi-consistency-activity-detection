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
    def __init__(self, name, clip_shape, batch_size, file_id, percent='40', use_random_start_frame=False):
      # self._dataset_dir = '/home/ke005409/Datasets/UCF101'      # CRCV cluster
      self._dataset_dir = '/home/akumar/dataset/UCF101'

      #self.get_det_annotations()       # To prepare pickle file for annots
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared(file_id, percent=percent)
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
      self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, file_id, percent='40'):
        import pickle
        
        training_annot_file = "../" + file_id
        
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        
        return training_annotations
        
        
    def get_det_annots_test_prepared(self):
        import pickle    
        with open('../testing_annots.pkl', 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)
            
        return testing_annotations    
    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        
        v_name, anns = self.vid_files[index]
        #print(v_name)
        clip, bbox_clip, label, annot_frames = self.load_video(v_name, anns)
        
        # Center crop
        frames, h, w, _ = clip.shape        
        margin_h = h - self._height
        h_crop_start = int(margin_h/2)
        margin_w = w - self._width
        w_crop_start = int(margin_w/2)
        
        clip = clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :] / 255.
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :]
        
        return clip, bbox_clip, label


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        # print(video_dir)
        # print(type(video_dir))
        try:
            # print(str(video_dir))
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
            # print(video.shape)
        except:
            # video = vread(str(video_dir), num_frames=40)
            print('Error:', str(video_dir))
            return None, None, None, None
            # print(video.shape)

        # video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        #if video.shape[0] < 40:
            #print(str(video_dir))
            #print(video.shape[0])

        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        #multi frame mode
        annot_idx = 0
        if len(annotations) > 1:
            annot_idx = np.random.randint(0,len(annotations))
        multi_frame_annot = []      # annotations[annot_idx][4]
        bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        #for ann in annotations:
        ann = annotations[annot_idx]    # For caps, use only 1 object at a time. 
        multi_frame_annot.extend(ann[4])
        start_frame, end_frame, label = ann[0], ann[1], ann[2]
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
        
        multi_frame_annot = list(set(multi_frame_annot))
        if self.name == 'train':
            return video, bbox_annot, label, multi_frame_annot
        else:
            return video, bbox, label, multi_frame_annot


if __name__ == '__main__':
    import imageio    
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    dataloader = UCF101DataLoader(name, clip_shape, batch_size, False)
    print(len(dataloader))
    #exit()
    index = 0
    while True:
        [clip, lbl_mask, cls_mask], [lbl, cls_lbl] = dataloader.__getitem__(index)
        #print(frm_idx)
        #pdb.set_trace()
        #with imageio.get_writer('./results/{:02d}_gt.gif'.format(index), mode='I') as writer:
        #    for i in range(clip.shape[1]):
        #        image = (clip[0,i]*255).astype(np.uint8)
        #        #image = image[:,:,::-1]
        #        writer.append_data(image) 
      
        for i in range(cls_lbl.shape[1]):
            for j in range(cls_lbl.shape[-1]):
                img = cls_lbl[0, i, :, :, j] * 255
                out_img = './results/samples/{:02d}_{:02d}_{:02d}_cls.jpg'.format(index, i, j)
                cv2.imwrite(out_img, img)
                
                img = cls_mask[0, i, :, :, j] * 255
                out_img = './results/samples/{:02d}_{:02d}_{:02d}_mask_cls.jpg'.format(index, i, j)
                cv2.imwrite(out_img, img)
            
            img = lbl[0, i, :, :, 0] * 255
            out_img = './results/samples/{:02d}_{:02d}_lbl.jpg'.format(index, i)
            cv2.imwrite(out_img, img)            
            
            img = lbl_mask[0, i, :, :, 0] * 255
            out_img = './results/samples/{:02d}_{:02d}_fg_mask_lbl.jpg'.format(index, i)
            cv2.imwrite(out_img, img)                        
                
        #out_img = './results/{:02d}_fg.jpg'.format(index)
        #img = lbl[0,0,:,:,0] * 255
        #cv2.imwrite(out_img, img)
      
      
        '''
        with imageio.get_writer('./results/{:02d}_lbl.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[1]):
              image = (lbl[0,i] * clip[0,i]).astype(np.uint8) 
              writer.append_data(image) 


        for j in range(mask.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mcls.gif'.format(index, j), mode='I') as writer:
          for i in range(mask.shape[1]):
            image = (mask[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
        '''

        '''
        for j in range(mask_mul.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mmul.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_mul.shape[1]):
            image = (mask_mul[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    

        for j in range(mask_add.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_madd.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_add.shape[1]):
            image = (mask_add[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
        '''
        print("Done for ", index)
        index += 1         
        exit() 
      
      
      
      








# def get_det_annotations(self):
#         print("Preparing train/test pickle...")
        
#         # f = loadmat(dataset_dir + 'UCF101_Annotations/trainAnnot.mat')
#         # f2 = loadmat(dataset_dir + 'UCF101_Annotations/testAnnot.mat')
#         f = loadmat(self._dataset_dir + '/trainAnnot.mat')
#         f2 = loadmat(self._dataset_dir + '/testAnnot.mat')

#         training_annotations = []
#         for ann in f['annot'][0]:
#             file_name = ann[1][0]

#             sp_annotations = ann[2][0]
#             annotations = []
            
#             for sp_ann in sp_annotations:
#                 frame_annot = []
#                 ef = sp_ann[0][0][0] - 1
#                 sf = sp_ann[1][0][0] - 1
#                 label = sp_ann[2][0][0] - 1
#                 bboxes = (sp_ann[3]).astype(np.int32)
#                 if ef - sf > 80:
#                     frames_to_choose = 5
#                 elif ef - sf >50:
#                     frames_to_choose = 3
#                 elif ef - sf > 30:
#                     frames_to_choose = 2
#                 else:
#                     frames_to_choose = 1
#                 for i in range(frames_to_choose):
#                     cf = np.random.randint(sf, ef)    #sf + int((ef - sf) / 2)
#                     if cf < 30:
#                         frame_annot.append(ef - 5)
#                     else:
#                         frame_annot.append(cf)
#                 annotations.append((sf, ef, label, bboxes, frame_annot))
#             training_annotations.append((file_name, annotations))
        
#         '''
#         with open('training_annots_multi.pkl','wb') as wid:
#             pickle.dump(training_annotations, wid, pickle.HIGHEST_PROTOCOL)
#         exit(0)
#         '''
        
#         testing_annotations = []
#         for ann in f2['annot'][0]:
#             file_name = ann[1][0]

#             sp_annotations = ann[2][0]
#             annotations = []
#             frame_annot = []
#             for sp_ann in sp_annotations:
#                 ef = sp_ann[0][0][0] - 1
#                 sf = sp_ann[1][0][0] - 1
#                 label = sp_ann[2][0][0] - 1
#                 bboxes = (sp_ann[3]).astype(np.int32)
#                 if len(annotations) == 0:
#                     cf = sf + int((ef - sf) / 2)
#                     if cf < 30:
#                         frame_annot.append(ef - 5)
#                     else:
#                         frame_annot.append(cf)
#                 annotations.append((sf, ef, label, bboxes, frame_annot))

#             testing_annotations.append((file_name, annotations))
#         '''
#         with open('testing_annots.pkl','wb') as wid2:
#             pickle.dump(testing_annotations, wid2, pickle.HIGHEST_PROTOCOL)
#         exit()
#         '''
#         return training_annotations, testing_annotations
