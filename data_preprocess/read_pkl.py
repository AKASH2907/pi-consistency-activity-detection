import pickle
import glob 
import random
import os

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

with open("training_annots.pkl", "rb") as trains:
	 train_annots = pickle.load(trains)

counter = 0


def data_percentage():
	total_files = 0
	with open('classes_list_25_per_class_random.txt', 'w+') as cf:
		for cls_list in classes_list:
			f = open(cls_list, 'r')
			wb = f.readlines()
			random.Random(47).shuffle(wb)

			n_files = int(0.25 *len(wb))
			total_files+= n_files
			# Minimum no of vids per class on 80% data
			for file in range(n_files):
				cf.write(wb[file])

			# break	
	print(total_files)
	

data_percentage()


def make_class_list():
	for i in range(len(train_annots)):
		# print(train_annots[i])
		anns = train_annots[i]
		clss = anns[0].split('/')[0]

		print(anns[0], clss)
		with open('classes_list/'+clss+'.txt', 'a+') as tv:
			tv.write(anns[0] + '\n')

classes_list = glob.glob("classes_list/*.txt")
classes_list.sort()
