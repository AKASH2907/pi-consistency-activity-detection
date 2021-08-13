import os
import random
import json

videos_list = os.listdir("/datasets/YouTube-VOS/2019/train/JPEGImages/")

print(len(videos_list))
# print(videos_list[:10])

random.Random(47).shuffle(videos_list)

# print(videos_list[:10])


all_anns = json.load(open("/datasets/YouTube-VOS/2019/train/meta.json", "r"))
# print(all_anns[0])
print(len(all_anns))  # 3471
# exit()
print(all_anns['videos'][videos_list[1]])


subset_20_len = int(0.2* len(videos_list))
print(subset_20_len)
subset_20_vids = videos_list[subset_20_len:]
print(len(subset_20_vids))
counter = 0

with open('meta_80percent_unlabeled.json', 'w') as f:
	d = {}
	d["videos"] = {}

	for i in range(len(subset_20_vids)):
		annotations = all_anns['videos'][videos_list[i]]

		d["videos"][videos_list[i]] = annotations
		# print(len(d), d)
		
		# exit()
		# if len(d)>1: exit()
		# if counter==2: break
		# counter += 1
	json.dump(d, f, indent=4)
	exit()
