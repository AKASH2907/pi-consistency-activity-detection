#from matplotlib import pyplot
import os
import config
import json
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.stats import norm

class TrainDataset(Dataset):
    def __init__(self, root='./data/train/', meta_root='./data/train/', json_file = 'meta_10percent_labeled.json'):
        self.train_frames = []
        self.train_annotations = []
        self.train_annotation_percent = []
        
        data_path = os.path.join(meta_root, json_file)
        #data_path = os.path.join(meta_root, 'meta_20per.json')
        with open(data_path) as f_train:
            train_data = json.load(f_train)
        print("Loaded data samples from: ", data_path)
        
        i = 0
        for video_folders in train_data['videos']:
            video_frames = []
            video_annotations = []
            video_percent_annotations = []
            
            for objects in train_data['videos'][video_folders]['objects']:
                object_frames = []
                object_annotations = []
                object_percent_annotations = []

                video_all_frames = os.listdir(os.path.join(root, 'JPEGImages/', video_folders + '/'))
                video_all_frames.sort()
                for frames in video_all_frames:
                    frame = os.path.join(root, 'JPEGImages/', video_folders + '/', frames)
                    object_frames.append(frame)
                
                # All annots
                for frames in train_data['videos'][video_folders]['objects'][objects]['frames']:
                    annotation = os.path.join(root, 'Annotations/', video_folders + '/', frames + '.png')
                    object_annotations.append(annotation)
                
                # Percentage annots 
                ##################
                # RIGHT NOW ALL FRAMES
                # for frames in train_data['videos'][video_folders]['objects'][objects]['frames_30p']:
                for frames in train_data['videos'][video_folders]['objects'][objects]['frames']:

                    annotation = os.path.join(root, 'Annotations/', video_folders + '/', frames + '.png')
                    object_percent_annotations.append(annotation)
                
                
                video_frames.append(object_frames)
                video_annotations.append((object_annotations))
                video_percent_annotations.append((object_percent_annotations))
            self.train_frames.append(video_frames)
            self.train_annotations.append(video_annotations)
            self.train_annotation_percent.append(video_percent_annotations)
            
            '''
            i += 1
            if i > 50:
                break 
            '''
            
        print("Data prep done for: ", len(self.train_frames))
        

    def __len__(self):
        return len(self.train_frames)


    def __getitem__(self, index):
        img_size_x = 224
        img_size_y = 224
        train_sample_frames = []#torch.zeros(32, 224, 224, 3)
        train_sample_annotations = []  # torch.zeros(1, 224, 224)
        train_sample_annotations.append([])
        train_sample_annotations_indeces = []#torch.zeros(224, 224, 3)
        train_sample_annotations_indeces.append([])

        train_sample_mask = np.zeros((config.n_frames, ))

        object = random.randint(0, len(self.train_frames[index])-1)
        rand_ann = len(self.train_annotation_percent[index][object]) - 1
        #rand_ann = len(self.train_annotation_percent[index][object]) - (config.n_anns+1)
        if rand_ann < 0:
            rand_ann = 0
        initial_annotation = random.randint(0,rand_ann)
        if initial_annotation < len(self.train_annotation_percent[index][object]):
            initial_frame_path = self.train_annotation_percent[index][object][initial_annotation].replace('Annotations', 'JPEGImages')
        else:
            print("Annot unavail for: ", len(self.train_annotation_percent[index][object]), " at ", initial_annotation)
            initial_frame_path = self.train_annotation_percent[index][object][0].replace('Annotations', 'JPEGImages')
        
        #initial_frame_path = self.train_annotations[index][object][initial_annotation].replace('Annotations', 'JPEGImages')
        initial_frame_path = initial_frame_path.replace('png', 'jpg')
        initial_frame = self.train_frames[index][object].index(initial_frame_path)
        
        #print("here")
        frames = initial_frame
        n_frames = config.n_frames
        frame_index = 0
        dropped_frames = 0
        max_dropped_frames = random.randint(0,4)
        skip_frames = False
        max_gaus = 0.29920671030107454
        #if random.random() < 0.15:
        #    n_frames *= 2
        #    skip_frames = True
        while frames < (initial_frame+n_frames):
            try:
                #print(frames)
                #print(self.train_frames[index][object][frames])
                frame = np.array(Image.open(self.train_frames[index][object][frames]))
                
                if(frame_index == 0):
                    vid_height, vid_width, _ = frame.shape
                '''
                if(vid_width >vid_height):
                    new_height_min = 0
                    new_height_max = vid_height
                    new_width_min = int(((vid_width/2) - (vid_height/2)))
                    new_width_max = int(((vid_width/2) + (vid_height/2)))
                elif(vid_width < vid_height):
                    new_height_min = int(((vid_height/2) - (vid_width/2)))
                    new_height_max =  int(((vid_height/2) + (vid_width/2)))
                    new_width_min = 0
                    new_width_max = vid_width
                else:
                    new_height_min = 0
                    new_height_max = vid_height
                    new_width_min = 0
                    new_width_max = vid_width
                '''
                annotation_path = self.train_frames[index][object][frames].replace('JPEGImages', 'Annotations')
                annotation_path = annotation_path.replace('jpg', 'png')
                if(annotation_path in self.train_annotations[index][object]):
                    annotation = np.array(Image.open(annotation_path))
                    annotation[annotation != (object + 1)] = 0
                    annotation = np.clip(annotation, 0, 1)
                    train_sample_annotations_indeces[0].append(frame_index)
                    if(annotation_path in self.train_annotation_percent[index][object]):
                        train_sample_mask[frame_index] = 1
                    else:
                        # Get gaussian val and update if max 
                        closest_frame_dist = np.min(np.abs(np.array(self.train_annotation_percent[index][object], np.int) - frames))
                        gaus_val = norm.pdf(closest_frame_dist/5., 0, 4/3.) / max_gaus
                        if train_sample_mask[frame_index] < gaus_val:
                            train_sample_mask[frame_index] = gaus_val
                            
                    #print(frames)
                else:
                    if random.random() < 0.08 and dropped_frames < max_dropped_frames and config.use_fixes:
                        frames += 1
                        n_frames += 1
                        dropped_frames += 1
                        continue
                    annotation = np.zeros((vid_height, vid_width))
                    train_sample_mask[frame_index] = 0

                #annotation = annotation[new_height_min:new_height_max, new_width_min:new_width_max]
                #frame = frame[new_height_min:new_height_max, new_width_min:new_width_max]

                frame = np.array(Image.fromarray(frame).resize(size=(img_size_x, img_size_y)))
                annotation = np.array(Image.fromarray(annotation).resize(size=(img_size_x, img_size_y)))

                train_sample_frames.append(frame)
                train_sample_annotations[0].append(annotation)
            except:
                train_sample_frames.append(train_sample_frames[-1])
                train_sample_annotations[0].append(train_sample_annotations[0][-1])

            if skip_frames:
                frames+=2
            else:
                frames+=1
            
            frame_index+=1

        
        while(len(train_sample_annotations_indeces[0]) < config.n_anns):
            train_sample_annotations_indeces[0].append(train_sample_annotations_indeces[0][-1])
        while (len(train_sample_annotations_indeces[0]) > config.n_anns):
            train_sample_annotations_indeces[0].pop()

        train_sample_frames = np.stack(train_sample_frames, 0)
        train_sample_frames = np.transpose(train_sample_frames, (3, 0, 1, 2))
        #print(train_sample_frames.shape)
        train_sample_annotations = np.stack(train_sample_annotations, 0)
        #print(train_sample_annotations.shape)
        train_sample_annotations_indeces = np.stack(train_sample_annotations_indeces, 0)
        #print(train_sample_annotations_indeces.shape)

        if np.all(train_sample_annotations[0][0] == 0) and config.use_fixes:
            train_sample_mask *= 0

        train_sample_mask = np.reshape(train_sample_mask, (1, config.n_frames, 1, 1))
        if config.skew_weight == True:
            train_sample_mask[:, 20:] *= 3 
        
        train_sample_frames = torch.from_numpy(train_sample_frames).type(torch.float)
        train_sample_annotations = torch.from_numpy(train_sample_annotations).type(torch.float)
        train_sample_annotations_indeces = torch.from_numpy(train_sample_annotations_indeces).type(torch.float)

        '''
        print_frames = train_sample_frames.cpu().detach().numpy().astype(np.uint8) #* 255
        for i in range(len(print_frames[0])):
            c = Image.fromarray(print_frames[0][i], mode='P')  # .resize(size=palette.size)
            # c.putpalette(palette.getpalette())
            c.save('./Print/%d_frame.png' % i, "PNG", mode='P')
        print_ann = train_sample_annotations.cpu().detach().numpy().astype(np.uint8)  * 255
        for i in range(len(print_ann[0])):
            c = Image.fromarray(print_ann[0][i], mode='P')  # .resize(size=palette.size)
            # c.putpalette(palette.getpalette())
            c.save('./Print/%d_ann.png' % i, "PNG", mode='P')
        '''
        
        return train_sample_frames, train_sample_annotations, train_sample_annotations_indeces, train_sample_mask

#seperate changes
class ValidationDataset(Dataset):
    def __init__(self, root='./data/valid/'):
        self.valid_frames = []
        self.valid_annotations = []

        with open(os.path.join(root, 'meta.json')) as f_valid:
            valid_data = json.load(f_valid)

        for video_folders in valid_data['videos']:
            video_frames = []
            video_annotations = []
            for objects in valid_data['videos'][video_folders]['objects']:
                object_frames = []
                object_annotations = []

                video_all_frames = os.listdir(os.path.join(root, 'JPEGImages/', video_folders + '/'))
                video_all_frames.sort()
                for frames in video_all_frames:
                    frame = os.path.join(root, 'JPEGImages/', video_folders + '/', frames)
                    object_frames.append(frame)
                for frames in valid_data['videos'][video_folders]['objects'][objects]['frames']:
                    #frame = os.path.join(root, 'valid/JPEGImages/', video_folders + '/', frames + '.jpg')
                    annotation = os.path.join(root, 'Annotations/', video_folders + '/', frames + '.png')
                    #object_frames.append(frame)
                    object_annotations.append(annotation)

                video_frames.append(object_frames)
                video_annotations.append((object_annotations))
            self.valid_frames.append(video_frames)
            self.valid_annotations.append(video_annotations)

    def __len__(self):
        return len(self.valid_frames)

    def __getitem__(self, index):
        img_size_x = 224
        img_size_y = 224
        valid_sample_frames = []    #torch.zeros(32, 224, 224, 3)
        valid_sample_annotations = []   #torch.zeros(1, 224, 224)
        valid_sample_annotations.append([])
        valid_sample_start_frame_indeces = []   #torch.zeros(224, 224, 3)
        valid_sample_start_frame_indeces.append([])

        for frames in range(len(self.valid_frames[index][0])):
            try:
                frame = np.array(Image.open(self.valid_frames[index][0][frames]))
                frame = np.array(Image.fromarray(frame).resize(size=(img_size_x, img_size_y)))

                valid_sample_frames.append(frame)
            except:
                valid_sample_frames.append(valid_sample_frames[-1])
        for objects in range(len(self.valid_frames[index])):
            annotation_path = self.valid_annotations[index][objects][0]
            annotation = np.array(Image.open(annotation_path))
            annotation[annotation != (objects + 1)] = 0
            annotation = np.clip(annotation, 0, 1)
            annotation = np.array(Image.fromarray(annotation).resize(size=(img_size_x, img_size_y)))

            initial_frame_path = annotation_path.replace('Annotations', 'JPEGImages')
            initial_frame_path = initial_frame_path.replace('png', 'jpg')
            initial_frame = self.valid_frames[index][objects].index(initial_frame_path)

            valid_sample_start_frame_indeces[0].append(initial_frame)
            valid_sample_annotations[0].append(annotation)

        valid_sample_frames = np.stack(valid_sample_frames, 0)
        valid_sample_frames = np.transpose(valid_sample_frames, (3, 0, 1, 2))
        #print(valid_sample_frames.shape)
        valid_sample_annotations = np.stack(valid_sample_annotations, 0)
        #print(valid_sample_annotations.shape)
        valid_sample_annotations_indeces = np.stack(valid_sample_start_frame_indeces, 0)
        #print(valid_sample_annotations_indeces.shape)

        valid_sample_frames = torch.from_numpy(valid_sample_frames).type(torch.float)
        valid_sample_annotations = torch.from_numpy(valid_sample_annotations).type(torch.float)
        valid_sample_annotations_indeces = torch.from_numpy(valid_sample_annotations_indeces).type(torch.float)
        return valid_sample_frames, valid_sample_annotations, valid_sample_annotations_indeces

if __name__ == '__main__':
    train = TrainDataset()
    x, y, z, _ = train.__getitem__(6)