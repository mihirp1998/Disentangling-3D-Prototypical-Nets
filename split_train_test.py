import ipdb
import os
import glob
join = os.path.join
st = ipdb.set_trace
glob_true = True
import socket
ip = socket.gethostname()
if "Alien" in ip:
	root_location = "/media/mihir/dataset/"
elif 'ip' in ip:
	root_location = "/projects/"
elif 'compute' in ip:
	root_location = "/home/mprabhud/dataset"

data_mod = 'rotMA500'
# folder_name = 'vqa_2_3_obj'
folder_name = 'vqa_2_3_obj_multi_azimuth'
split_percent =  0.25
split_percent_c  = 1 - split_percent
name_split = f"{split_percent}_{split_percent_c}"
if glob_true:
	location = f"{root_location}/vqa/{folder_name}/npy/{data_mod}/*"
	files = glob.glob(location)
	files = ['/'.join(i.split('/')[-2:]) +"\n" for i in files]
	num = int(len(files)*split_percent)
	txt_file_train_1 =  f"{root_location}/vqa/{folder_name}/npy/{data_mod}_{name_split}t.txt"
	txt_file_train_2 =  f"{root_location}/vqa/{folder_name}/npy/{data_mod}_{name_split}v.txt"
	train_data_1 = files[:num]
	train_data_2 = files[num:]
else:
	data_mod = "bb_tv"
	new_mod_1 = data_mod +"_a"
	new_mod_2 = data_mod +"_b"
	root_location = "/projects/katefgroup/datasets/"
	txt_file_train = f"{root_location}/carla/npys/{data_mod}t.txt"

	txt_file_train_1 = f"{root_location}/carla/npys/{new_mod_1}t.txt"
	txt_file_train_2 = f"{root_location}/carla/npys/{new_mod_2}t.txt"

	train_data = open(txt_file_train,"r").readlines()
	num = int(len(train_data)*split_percent)
	train_data_1 = train_data[:num]
	train_data_2 = train_data[num:]


with open(txt_file_train_1, 'w') as f:
	for item in train_data_1:
		if "*" not in item:
			f.write("%s" % item)

with open(txt_file_train_2, 'w') as f:
	for item in train_data_2:
		if "*" not in item:
			f.write("%s" % item)