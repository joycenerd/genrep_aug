from pathlib import Path
import numpy as np
import random
import shutil
import os


filename='../data/imagenet_folder.txt'
f=open(filename,'r')
dir_list=[]
for dir_name in f:
    dir_name=dir_name.strip()
    if len(dir_name)>0:
        dir_list.append(dir_name)
# print(dir_list)

src_dir='/eva_data/zchin/steer_img/biggan-deep-256_tr1.0_steer_pth_imagenet100_N20/train'
dest_dir='../data/gen_train1300'
if not os.path.isdir(dest_dir):
    os.makedirs(dest_dir)

for dir_name in dir_list:
    src_path=Path(src_dir).joinpath(dir_name)
    dest_path=Path(dest_dir).joinpath(dir_name)
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    
    img_dict={}
    idx=0
    for _file in os.listdir(src_path):
        if _file[-4:]=='.png':
            img_dict[idx]=_file
            idx+=1
    
    rand_num=random.sample(range(idx),1300)
    for num in rand_num:
        src_file=Path(src_path).joinpath(img_dict[num])
        dest_file=Path(dest_path).joinpath(img_dict[num])
        shutil.copyfile(src_file,dest_file)
    
    print(f"{dir_name} complete...")

