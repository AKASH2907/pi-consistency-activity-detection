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
    def __init__(self, name, clip_shape, batch_size, percent='40', use_random_start_frame=False):
      # self._dataset_dir = '/home/ke005409/Datasets/UCF101'      # CRCV cluster
      self._dataset_dir = '/home/akumar/dataset/UCF101'
      #self.get_det_annotations()       # To prepare pickle file for annots
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared(percent=percent)
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
            

    def get_det_annots_prepared(self, percent='40'):
        import pickle
        
        '''
        #training_annot_file = 'training_annots.pkl'
        #training_annot_file = 'training_annots_multi.pkl'
        #training_annot_file = 'training_annots_multi_20perRand.pkl'
        #training_annot_file = 'training_annots_multi_20perEqui.pkl'
        #training_annot_file = 'training_annots_multi_5perRand.pkl'
        #training_annot_file = 'training_annots_multi_5perEqui.pkl'
        #training_annot_file = 'training_annots_multi_5perEqui_prune1.pkl'
        #training_annot_file = 'training_annots_multi_5perEqui_prune2.pkl'
        #training_annot_file = 'eval_5perEqui_prune2_annots.pkl'
        #training_annot_file = 'training_annots_multi_40perRand.pkl'
        #training_annot_file = 'training_annots_multi_30perRand.pkl'
        #training_annot_file = 'training_annots_multi_50perEqui.pkl'
        #training_annot_file = 'training_annots_multi_75perEqui.pkl'
        '''
        
        training_annot_file = 'training_annots.pkl'
        #training_annot_file = 'training_annots_fullAnnot_'+percent+'perCls.pkl'
        # training_annot_file = 'training_annots_multi_'+percent+'perRand.pkl'
        # training_annot_file = "../train_annots_20_per.pkl"
        # training_annot_file = "../updated_label_annots_iter_1.pkl"
        
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        
        return training_annotations
        
        
    def get_det_annots_test_prepared(self):
        import pickle    
        with open('../testing_annots.pkl', 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)
            
        return testing_annotations    
    
    
    def get_det_annotations(self):
        print("Preparing train/test pickle...")
        
        # f = loadmat(dataset_dir + 'UCF101_Annotations/trainAnnot.mat')
        # f2 = loadmat(dataset_dir + 'UCF101_Annotations/testAnnot.mat')
        f = loadmat(self._dataset_dir + '/trainAnnot.mat')
        f2 = loadmat(self._dataset_dir + '/testAnnot.mat')

        training_annotations = []
        for ann in f['annot'][0]:
            file_name = ann[1][0]

            sp_annotations = ann[2][0]
            annotations = []
            
            for sp_ann in sp_annotations:
                frame_annot = []
                ef = sp_ann[0][0][0] - 1
                sf = sp_ann[1][0][0] - 1
                label = sp_ann[2][0][0] - 1
                bboxes = (sp_ann[3]).astype(np.int32)
                if ef - sf > 80:
                    frames_to_choose = 5
                elif ef - sf >50:
                    frames_to_choose = 3
                elif ef - sf > 30:
                    frames_to_choose = 2
                else:
                    frames_to_choose = 1
                for i in range(frames_to_choose):
                    cf = np.random.randint(sf, ef)    #sf + int((ef - sf) / 2)
                    if cf < 30:
                        frame_annot.append(ef - 5)
                    else:
                        frame_annot.append(cf)
                annotations.append((sf, ef, label, bboxes, frame_annot))
            training_annotations.append((file_name, annotations))
        
        '''
        with open('training_annots_multi.pkl','wb') as wid:
            pickle.dump(training_annotations, wid, pickle.HIGHEST_PROTOCOL)
        exit(0)
        '''
        
        testing_annotations = []
        for ann in f2['annot'][0]:
            file_name = ann[1][0]

            sp_annotations = ann[2][0]
            annotations = []
            frame_annot = []
            for sp_ann in sp_annotations:
                ef = sp_ann[0][0][0] - 1
                sf = sp_ann[1][0][0] - 1
                label = sp_ann[2][0][0] - 1
                bboxes = (sp_ann[3]).astype(np.int32)
                if len(annotations) == 0:
                    cf = sf + int((ef - sf) / 2)
                    if cf < 30:
                        frame_annot.append(ef - 5)
                    else:
                        frame_annot.append(cf)
                annotations.append((sf, ef, label, bboxes, frame_annot))

            testing_annotations.append((file_name, annotations))
        '''
        with open('testing_annots.pkl','wb') as wid2:
            pickle.dump(testing_annotations, wid2, pickle.HIGHEST_PROTOCOL)
        exit()
        '''
        return training_annotations, testing_annotations

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        depth = 8
        video_rgb = np.zeros((depth, self._height, self._width, 3))     # 8, 224,224, 3
        label_cls = np.zeros((depth, self._height, self._width, 1))     # FG/BG or actor (person only) for this dataset
        
        v_name, anns = self.vid_files[index]
        #print(v_name)
        clip, bbox_clip, label, annot_frames = self.load_video(v_name, anns)
        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0])}
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
                sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0])}
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
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0])}
            return sample
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)
        
        #Random start frame
        if self._use_random_start_frame:
            random_start_frame_btm = selected_annot_frame - (depth * vskip) + 2
            if random_start_frame_btm < 0:
                random_start_frame_btm = 0
            random_start_frame_top = selected_annot_frame - 2
            if random_start_frame_top <= random_start_frame_btm:
                random_start_frame = start_frame
            else:
                random_start_frame = np.random.randint(random_start_frame_btm, random_start_frame_top)
            if random_start_frame + (depth * vskip) >= vlen:
                random_start_frame = vlen - (depth * vskip)
            start_frame = random_start_frame
        
        span = (np.arange(depth)*vskip)
        span += start_frame
        video = clip[span]
        bbox_clip = bbox_clip[span]
        closest_fidx = np.argmin(np.abs(span-selected_annot_frame))
        
        if self.name == 'train':
            #start_pos = np.random.randint(0,16)
            start_pos_h = np.random.randint(0,clip_h - 224) #self._height)
            start_pos_w = np.random.randint(0,clip_w - 224) #self._width)
        else:
            # center crop for validation
            start_pos_h = int((clip_h - 224) / 2)
            start_pos_w = int((clip_w - 224) / 2)
        
        '''
        # Masking version
        for ann in annot_frames:
            if span[0] <= ann <= span[-1]:
                closest_ann_idx = np.argmin(np.abs(span-ann))
                closest_ann_idx = closest_ann_idx // 2
                #print(ann, ', ', closest_ann_idx)
                batch_mask_cls_mul[i, closest_ann_idx, :,:,:] = 1.
                batch_label_cls_mul[i, closest_ann_idx, :,:,:] = 1.
                
                bbox_img = bbox_clip[ann]
                bbox_img = bbox_img[start_pos:start_pos+224, start_pos:start_pos+296,:]
                bbox_img = cv2.resize(bbox_img, (56,56), interpolation=cv2.INTER_NEAREST)
                bbox_img[bbox_img>0] = 1.
                batch_label_cls[i, closest_ann_idx, :, :, 0] = bbox_img
                batch_mask_cls[i, closest_ann_idx, bbox_img > 0, label] = 1.
                batch_mask_cls[i, closest_ann_idx, bbox_img > 0, -1] = 0.
        '''
                    
        '''
        # For Frame Pooling Version
        #start_pos = np.random.randint(0,16)
        bbox_img = bbox_clip[selected_annot_frame]
        bbox_img = bbox_img[start_pos:start_pos+224, start_pos:start_pos+296,:]
        bbox_img = cv2.resize(bbox_img, (56,56), interpolation=cv2.INTER_NEAREST)
        batch_label_cls_single_frame[i, 0, :, :, 0] = bbox_img
        batch_mask_cls_single_frame[i, 0, bbox_img > 0, label] = 1.
        batch_mask_cls_single_frame[i, 0, bbox_img > 0, -1] = 0.
        batch_frame_idx[i, 0] = int(closest_fidx/2)
        bbox_clip = bbox_clip[span]
        '''
        
        for j in range(video.shape[0]):
            img = video[j]
            #img = img[start_pos:start_pos+224, start_pos:start_pos+296,:]
            img = img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            img = cv2.resize(img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            img = img / 255.
            video_rgb[j] = img
            
            bbox_img = bbox_clip[j]
            #bbox_img = bbox_img[start_pos:start_pos+224, start_pos:start_pos+296,:]
            bbox_img = bbox_img[start_pos_h:start_pos_h+224, start_pos_w:start_pos_w+224,:]
            #bbox_img = bbox_img.reshape(self._height, self._width)
            bbox_img = cv2.resize(bbox_img, (self._height,self._width), interpolation=cv2.INTER_LINEAR)
            #lbl_idx = int(j/2)  # Overwrite for every 2 frames.. lame but workaround
            label_cls[j, bbox_img > 0, 0] = 1.
                       
        
        if np.random.randint(0,2) == 1:
            video_rgb[:,:,:,:] = video_rgb[:,:,::-1,:]
            label_cls[:,:,:,:] = label_cls[:,:,::-1,:]
        
        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)
        
        action_tensor = torch.Tensor([label])        
        #pdb.set_trace()
        sample = {'data':video_rgb,'segmentation':label_cls,'action':action_tensor}
        return sample


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
        for ann in annotations:
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
            # Expect to have collect_annots with same length as annots for this set 
            # [ c, c, c, c, c, c ....]
            select_annots = ann[4]
            select_annots.sort()
            if len(collect_annots) == 0:
                continue
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
            #return video, bbox_annot, label, multi_frame_annot
            return video, bbox, label, multi_frame_annot
        else:
            return video, bbox, label, multi_frame_annot


if __name__ == '__main__':
    import imageio    
    name='test'
    clip_shape=[112,112]
    channels=3
    batch_size = 1
    dataloader = UCF101DataLoader(name, clip_shape, batch_size, False)
    print(len(dataloader))
    #exit()
    index = 0
    while True:
        data = dataloader.__getitem__(index)
        print(data['action'])
        print(data['data'].shape)
        print(data['segmentation'].shape)
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
      
      
      
      
