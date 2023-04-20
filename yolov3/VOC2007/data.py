from genericpath import exists
import os
import shutil

save_dir_img_train = "E:/dataset/infrared/Main/images/train/"
save_dir_img_val = "E:/dataset/infrared/Main/images/val/"
save_dir_txt_train = "E:/dataset/infrared/Main/labels/train/"
save_dir_txt_val = "E:/dataset/infrared/Main/labels/val/"


if not os.path.exists(save_dir_img_train):
    os.makedirs(save_dir_img_train)
if not os.path.exists(save_dir_img_val):
    os.makedirs(save_dir_img_val)
if not os.path.exists(save_dir_txt_train):
    os.makedirs(save_dir_txt_train)
if not os.path.exists(save_dir_txt_val):
    os.makedirs(save_dir_txt_val)

fr_train = open("E:/dataset/infrared/Main/train.txt")
fr_val = open("E:/dataset/infrared/Main/test.txt")

dir_img = "E:/dataset/infrared/images/"
dir_txt = "E:/dataset/infrared/labels/"

for line in fr_train.readlines():
    line = line.strip()
    from_path_img_train = dir_img + str(line) + ".tif"
    from_path_txt_train = dir_txt + str(line) + ".txt"
    to_path_img_train = save_dir_img_train + str(line) + ".tif"
    to_path_txt_train = save_dir_txt_train + str(line) + ".txt"
    shutil.copyfile(from_path_img_train, to_path_img_train)
    shutil.copyfile(from_path_txt_train, to_path_txt_train)

for line in fr_val.readlines():
    line = line.strip()
    from_path_img_val = dir_img + str(line) + ".tif"
    from_path_txt_val = dir_txt + str(line) + ".txt"
    to_path_img_val = save_dir_img_val + str(line) + ".tif"
    to_path_txt_val = save_dir_txt_val + str(line) + ".txt"
    shutil.copyfile(from_path_img_val, to_path_img_val)
    shutil.copyfile(from_path_txt_val, to_path_txt_val)
    