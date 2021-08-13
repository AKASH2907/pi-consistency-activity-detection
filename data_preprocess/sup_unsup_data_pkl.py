import pickle
import glob 
import random
import cv2
import os
import numpy as np

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

with open("training_annots.pkl", "rb") as trains:
	 train_annots = pickle.load(trains)

counter = 0

def ucf101_all_action_classes():
	video_path = "/datasets/UCF-101/TrainingData"
	action_recog_train_split = open("/datasets/UCF-101/ActionRecognitionSplits/trainlist01.txt", "r").readlines()

	# print(len(action_recog_train_split))
	# print(action_recog_train_split[0])

	action_detect = list()
	action_recog = list()

	for i in range(len(train_annots)):
		action_detect.append(train_annots[i][0])
	print(len(action_detect))
	# print(action_detect[:50])
	for i in range(len(action_recog_train_split)):
		action_recog.append(action_recog_train_split[i].split(" ")[0].split(".")[0])
	print(len(action_recog))
	# print(action_recog[:50])
	# def ucf101_remaining_actions_pkl_file():

	list_diff = list(set(action_recog) - set(action_detect))
	print(len(list_diff))
	list_diff = sorted(list_diff)
	# print(list_diff[:50])

	'''0

	go thru labeled list all 
	'''

	# wr_filename_1 = open('train_annots_24_actions_labeled.pkl', 'wb')
	# wr_filename_2 = open('train_annots_77_actions_unlabeled.pkl', 'wb')

	train_data = []

	lbl_counter = 0
	unlbl_counter = 0
	for x in range(len(action_recog)):

		if action_recog[x] not in action_detect:
			# print(action_recog[x])
			# print(action_detect[lbl_counter])
			anns = train_annots[lbl_counter] 
			
			new_ann = ()
			new_ann += (action_recog[x],)
			vid_info = list()

			strt_frm = 0
			cap = cv2.VideoCapture(os.path.join(video_path, action_recog[x] + '.avi'))
			num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			end_frm = num_frames - 1
			# print(end_frm)
			vid_infos = ()
			vid_infos += (int(strt_frm),)
			vid_infos += (int(end_frm),)
			vid_infos += (-1,)
			# print(np.zeros((1, 4), dtype=np.int32))
			vid_infos += (np.zeros((1, 4), dtype=np.int32),)
			frames = np.arange(int(num_frames)).tolist()
			vid_infos += (frames,)
			vid_infos += (0,)
			vid_info.append(vid_infos)
			# print(vid_infos)
			new_ann += (vid_info,)
			# print(vid_info)
			# print(new_ann)
			train_data.append(new_ann)
			unlbl_counter += 1

		else:
			for n_ann in range(len(anns[1])):
				anns[1][n_ann] += (1,)
			train_data.append(anns)
			lbl_counter +=1

			# exit()

	print(len(train_data))
	# print(train_data)
	# print(unlbl_counter)
	# pickle.dump(train_data, wr_filename_1)
	# print(lbl_counter)



def make_sup_unsup_percent_pkl_files():
	# percent_data = open('classes_list_20_per.txt')
	percent_data = open('classes_list_25_per_class_random.txt')
	pd = percent_data.readlines()
	train_data = []

	# train
	# wr_filename = open('train_annots_80_unlabeled_57_vids.pkl', 'wb')
	# wr_filename = open('training_annots_with_labels.pkl', 'wb')
	wr_filename = open('train_annots_25_unlabeled_random.pkl', 'wb')
	# test
	# wr_filename = open('test_annots.pkl', 'wb')
	
	counter = 0
	for x in range(len(train_annots)):
		anns = train_annots[x]
		if anns[0]+ '\n' in pd:
			for n_ann in range(len(anns[1])):
				# print(anns[1][n_ann])
				# anns[1][n_ann].append((1,))
				anns[1][n_ann] +=(1,)

			train_data.append(anns)

	print(len(train_data))
	# exit()
	pickle.dump(train_data, wr_filename)
	wr_filename.close()

# make_sup_unsup_percent_pkl_files()


def pkl_file_check_before_train():

	with open("train_annots_75_unlabeled_random.pkl", "rb") as read_file:
		check_annots = pickle.load(read_file)

		print(check_annots[3])
# pkl_file_check_before_train()

def make_data_percent_pkl_files():
	percent_data = open('classes_list_20_per.txt')

	percent_data = open('classes_list_5_per.txt')
	pd = percent_data.readlines()

	train_data = []
	
	# train
	wr_filename = open('train_annots_80_unlabeled_data.pkl', 'wb')

	for x in range(len(train_annots)):
		anns = train_annots[x]
		if anns[0] + '\n' not in pd:
			# print(anns[1])
			# if 'sup_binary_index' not in 
			train_data.append(anns)

	pickle.dump(train_data, wr_filename)
	print(len(train_data))
	wr_filename.close()

# make_data_percent_pkl_files()