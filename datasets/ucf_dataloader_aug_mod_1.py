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
          self.vid_files = self.get_det_annots_test_prepared(file_id)
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
        
        training_annot_file = "../"+ file_id
        
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
        # print(v_name)
        clip, bbox_clip, label, annot_frames, labeled_vid = self.load_video(v_name, anns)
        # print(f'vlen, clip_h, clip_w, _: {clip.shape}')
        # print(f'Bbox shape:  {bbox_clip.shape}')
        # print(f'Annotated frames: {annot_frames}')
        # print(f'Video label: {label}') 
        # labeled_vid)
        # exit()
        if clip is None:
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'label_vid': labeled_vid}
            # sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'aug_label': label_cls, 'label_vid': labeled_vid, 'aug_type': torch.Tensor([0])}

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
                # print("---------NOBBOXERR-----------")
                sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'label_vid': labeled_vid}
                # sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'aug_label': label_cls, 'label_vid': labeled_vid, 'aug_type': torch.Tensor([0])}

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
            # print("-------NOBBOXERR-----------")
            video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
            video_rgb = torch.from_numpy(video_rgb)            
            label_cls = np.transpose(label_cls, [3, 0, 1, 2])
            label_cls = torch.from_numpy(label_cls)
            sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'label_vid': labeled_vid}

            # sample = {'data':video_rgb,'segmentation':label_cls,'action':torch.Tensor([0]), 'aug_data': video_rgb, 'aug_label': label_cls, 'label_vid': labeled_vid, 'aug_type': torch.Tensor([0])}
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
        # print(span)
        span += start_frame
        # print(span)
        video = clip[span]
        # print(video.shape)
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
        
        # print(video.shape)
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
                       
        # print(video_rgb.shape)
        # if np.random.randint(0,2) == 1:
        # print(video_rgb[0, 0, 0, :].shape)
        # video_rgb[:,:,:,:] = video_rgb[:,:,::-1,:]
        # label_cls[:,:,:,:] = label_cls[:,:,::-1,:]
        # print(video_rgb.shape)
        # exit()
        # VIDEO RGB has been cropped -> 224, 224
        # HORIZONTAL FLIPPING        

        # random_aug = torch.randint(1, 4, (1,))
        # print(random_aug, type(random_aug))
        # random_aug = 4
        # print(video_rgb.shape, label_cls.shape)
        # exit()
        # print("**************")
        # print(video_rgb.shape, label_cls.shape)

        # anticlockwise rotation with np.rot90
        # if random_aug == 1:

        #     # horizontal_flipped_video = video_rgb[:, :, ::-1, :]
        #     # horizontal_flipped_label_cls = label_cls[:,:,::-1,:]

        #     aug_video = video_rgb[:, :, ::-1, :]
        #     aug_label_cls = label_cls[:,:,::-1,:]
        #     # print(aug_video.shape, aug_label_cls.shape)
        # elif random_aug == 2:  #270 degrees rotation
        #     # print(video_rgb.shape, label_cls.shape)
        #     aug_video = np.rot90(video_rgb, 1, (1, 2))
        #     aug_label_cls = np.rot90(label_cls, 1, (1, 2))
        #     # print(aug_video.shape, aug_label_cls.shape)

        # elif random_aug == 3:  #90 degrees rotation
        #     # print(video_rgb.shape, label_cls.shape)
        #     aug_video = np.rot90(video_rgb, 3, (1, 2))
        #     aug_label_cls = np.rot90(label_cls, 3, (1, 2))


        # aug_video = np.transpose(horizontal_flipped_video, [3, 0, 1, 2])
        # aug_video = torch.from_numpy(horizontal_flipped_video.copy())
        # aug_label_cls = np.transpose(horizontal_flipped_label_cls, [3, 0, 1, 2])
        # aug_label_cls = torch.from_numpy(horizontal_flipped_label_cls.copy())
        video_rgb = np.transpose(video_rgb, [3, 0, 1, 2])  #moving channels to first position
        video_rgb = torch.from_numpy(video_rgb)
        label_cls = np.transpose(label_cls, [3, 0, 1, 2])
        label_cls = torch.from_numpy(label_cls)

        # aug_video = np.transpose(aug_video, [3, 0, 1, 2])
        # aug_video = torch.from_numpy(aug_video.copy())
        # aug_label_cls = np.transpose(aug_label_cls, [3, 0, 1, 2])
        # aug_label_cls = torch.from_numpy(aug_label_cls.copy())

        
        action_tensor = torch.Tensor([label])   
        vid_lab_unlab = torch.Tensor([labeled_vid])
        # print(v_name, vid_lab_unlab)
        # print(len(vid_lab_unlab))
        # if len(vid_lab_unlab)<1: print(v_name)
        # exit()
        #pdb.set_trace()
        # sample = {'data':video_rgb,'segmentation':label_cls,'action':action_tensor, 'video_label_avail': vid_lab_unlab}
        sample = {'data':video_rgb,'segmentation':label_cls,'action':action_tensor,'label_vid': labeled_vid}

        # sample = {'data':video_rgb,'segmentation':label_cls,'action':action_tensor, "aug_data":aug_video, "aug_label": aug_label_cls, 'label_vid': labeled_vid, 'aug_type': random_aug}
        # if 'video_label_avail' in sample:
            # print("true")
            # exit()
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
        # print(len(annotations))
        # print(annotations[5])
        # exit()
        for ann in annotations:
            # print("*********",ann[4])
            if len(ann)<5:
                print("video_name:", video_name)
            # labeled_vid = ann[5]
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
            return video, bbox, label, multi_frame_annot, labeled_vid
        else:
            return video, bbox, label, multi_frame_annot, labeled_vid

    # def random_flip(self, buffer):
    #     "Horizontally flip with probability of 0.5"
    #     if np.random.random()<0.5:
    #         for 


if __name__ == '__main__':
    import imageio    
    name='test'
    clip_shape=[112,112]
    channels=3
    batch_size = 1
    # dataloader = UCF101DataLoader(name, clip_shape, batch_size, False)
    dataloader = UCF101DataLoader('train', [224, 224], batch_size=4, file_id='../train_annots_20_labeled.pkl', percent='100', use_random_start_frame=False)

    print(len(dataloader))
    #exit()
    index = 0
    while True:
        data = dataloader.__getitem__(index)
        # print(data['action'])
        # print(data['data'].shape)
        # print(data['segmentation'].shape)

        # exit()

        clip = data['data']
        clip = clip.numpy()
        clip = np.transpose(clip, [1, 2, 3, 0])

        clip_mask = data['segmentation']
        clip_mask = clip_mask.numpy()
        clip_mask = np.transpose(clip_mask, [1, 2, 3, 0])

        flipped_clip = data['aug_data']
        flipped_clip = flipped_clip.numpy()
        flipped_clip = np.transpose(flipped_clip, [1, 2, 3, 0])

        flip_mask = data['aug_label']
        flip_mask = flip_mask.numpy()
        flip_mask = np.transpose(flip_mask, [1, 2, 3, 0])

        # diff2 = torch.sub(data['segmentation'], data['aug_label'])
        # diff2 = diff2.numpy()
        # diff2 = np.transpose(diff2, [1, 2, 3, 0])
        # print(clip_mask.shape, flip_mask.shape, np.sum(clip_mask), np.sum(flip_mask), np.unique(clip_mask))
        # print(clip_mask[0,:, :, 0])

        #print(frm_idx)
        #pdb.set_trace()

        # difference = np.abs(clip_mask,flip_mask) 
        # difference2 = np.abs(flip_mask, clip_mask)

        # print(np.sum(difference))
        # from scipy.spatial.distance import cdist
        # s1 = clip_mask[0, :, :, 0]
        # s2 = flip_mask[0, :, :, 0]
        # print(s1.shape, s2.shape, np.sum(s1), np.sum(s2))
        # print(s1.max())
        # l2 = cdist(s1, s2, 'minkowski', p=1.)
        # l2 = np.abs(s1, s2) + np.abs(s2, s1)
        # print(np.sum(l2), np.unique(l2))
        # print(np.where(l2==1))
        # exit()
        # print(np.unique(diff2))
        # diff2 = np.abs(diff2)
        # print(np.sum(diff2))
        # corrected_mask = flip_mask[:, :, ::-1,:]
        # diff = np.subtract(clip_mask, flip_mask)
        

        # diff = np.abs(diff)

        # print(np.unique(diff), np.sum(diff))
        # with imageio.get_writer('./vis_dataloader/diff_{:02d}_gt.gif'.format(index), mode='I') as writer:
        #     for i in range(diff.shape[0]):
        #         image = (diff[i]*255).astype(np.uint8)
        #         writer.append_data(image) 
        # with imageio.get_writer('./results/diff2_{:02d}_gt.gif'.format(index), mode='I') as writer:
        #     for i in range(diff2.shape[0]):
        #         image = (diff2[i]*255).astype(np.uint8)
        #         writer.append_data(image) 
        # exit()
        with imageio.get_writer('./vis_dataloader/rand_orig_mask_based_{:02d}_gt.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[0]):
                image = (clip[i]*255).astype(np.uint8)
                writer.append_data(image) 
        with imageio.get_writer('./vis_dataloader/rand_aug_mask_based_{:02d}_gt.gif'.format(index), mode='I') as writer:
            for i in range(flipped_clip.shape[0]):
                image = (flipped_clip[i]*255).astype(np.uint8)
                writer.append_data(image)


        with imageio.get_writer('./vis_dataloader/rand_aug_mask_{:02d}_gt.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[0]):
                image = (clip[i,:,:,0]*255).astype(np.uint8)
                cl_mask = (clip_mask[i,:,:,0]*255).astype(np.uint8)
                # print(image.shape,clip_mask[i,:,:,0].shape)
                # image = cv2.drawContours(image, clip_mask[i,:,:,0][0], -1, (0 , 255, 0), 3)
                image = cv2.bitwise_and(image, image, mask=cl_mask)
                writer.append_data(image) 

        with imageio.get_writer('./vis_dataloader/rand_aug_mask_{:02d}_gt.gif'.format(index), mode='I') as writer:
            for i in range(flipped_clip.shape[0]):
                image = (flipped_clip[i]*255).astype(np.uint8)
                fl_mask = (flip_mask[i,:,:,0]*255).astype(np.uint8)

                image = cv2.bitwise_and(image, image, mask=fl_mask)

                writer.append_data(image)
        
        # cls_lbl = data['segmentation']
        # print(cls_lbl.shape[-1])

        # for 


        # for i in range(clip.shape[0]):
        #     for j in range(clip.shape[-1]):
        #         img = clip[i, :, :, j] * 255
        #         print(img.shape)
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
                
        #         img = cls_mask[i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_mask_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
            
        #     img = lbl[i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)            
            
        #     img = lbl_mask[i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_fg_mask_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)  

        # for i in range(cls_lbl.shape[1]):
        #     for j in range(cls_lbl.shape[-1]):
        #         img = cls_lbl[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
                
        #         img = cls_mask[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_mask_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
            
        #     img = lbl[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)            
            
        #     img = lbl_mask[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_fg_mask_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)                        
                
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
#     print("Preparing train/test pickle...")
    
#     # f = loadmat(dataset_dir + 'UCF101_Annotations/trainAnnot.mat')
#     # f2 = loadmat(dataset_dir + 'UCF101_Annotations/testAnnot.mat')
#     f = loadmat(self._dataset_dir + '/trainAnnot.mat')
#     f2 = loadmat(self._dataset_dir + '/testAnnot.mat')

#     training_annotations = []
#     for ann in f['annot'][0]:
#         file_name = ann[1][0]

#         sp_annotations = ann[2][0]
#         annotations = []
        
#         for sp_ann in sp_annotations:
#             frame_annot = []
#             ef = sp_ann[0][0][0] - 1
#             sf = sp_ann[1][0][0] - 1
#             label = sp_ann[2][0][0] - 1
#             bboxes = (sp_ann[3]).astype(np.int32)
#             if ef - sf > 80:
#                 frames_to_choose = 5
#             elif ef - sf >50:
#                 frames_to_choose = 3
#             elif ef - sf > 30:
#                 frames_to_choose = 2
#             else:
#                 frames_to_choose = 1
#             for i in range(frames_to_choose):
#                 cf = np.random.randint(sf, ef)    #sf + int((ef - sf) / 2)
#                 if cf < 30:
#                     frame_annot.append(ef - 5)
#                 else:
#                     frame_annot.append(cf)
#             annotations.append((sf, ef, label, bboxes, frame_annot))
#         training_annotations.append((file_name, annotations))
    
#     '''
#     with open('training_annots_multi.pkl','wb') as wid:
#         pickle.dump(training_annotations, wid, pickle.HIGHEST_PROTOCOL)
#     exit(0)
#     '''
    
#     testing_annotations = []
#     for ann in f2['annot'][0]:
#         file_name = ann[1][0]

#         sp_annotations = ann[2][0]
#         annotations = []
#         frame_annot = []
#         for sp_ann in sp_annotations:
#             ef = sp_ann[0][0][0] - 1
#             sf = sp_ann[1][0][0] - 1
#             label = sp_ann[2][0][0] - 1
#             bboxes = (sp_ann[3]).astype(np.int32)
#             if len(annotations) == 0:
#                 cf = sf + int((ef - sf) / 2)
#                 if cf < 30:
#                     frame_annot.append(ef - 5)
#                 else:
#                     frame_annot.append(cf)
#             annotations.append((sf, ef, label, bboxes, frame_annot))

#         testing_annotations.append((file_name, annotations))
#     '''
#     with open('testing_annots.pkl','wb') as wid2:
#         pickle.dump(testing_annotations, wid2, pickle.HIGHEST_PROTOCOL)
#     exit()
#     '''
#     return training_annotations, testing_annotations