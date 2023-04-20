import cv2
import numpy as np
import os

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0

# image = cv2.imread(r'E:\code\shaped_mask_attack\yolov3\VOC2007\attackImageSets\2014_05_04_23_14_54_735000.tif')

# put your image(tif) to be cropped there. you'd better find some images with big objects. dont rename!
crop_img_ori_dir = r"E:\code\shaped_mask_attack\yolov3\VOC2007\images\train"
# put all annotation here
anno_dir = r"E:\code\shaped_mask_attack\yolov3\VOC2007\label\train"
# your cropped img result will be saved here
img_save_dir = r"E:\code\shaped_mask_attack\yolov3\VOC2007\images\attack"
# your annotation of the same name with img will be saved here
anno_save_dir = r"E:\code\shaped_mask_attack\yolov3\VOC2007\label\attack"
img_path_list = os.listdir(crop_img_ori_dir)
cnt = 0

def next_img():
    img_path = crop_img_ori_dir + "/" + img_path_list[cnt]
    image = cv2.imread(img_path)
    if cnt == 0:
        anno_name = img_path_list[cnt].split(".")[0] + ".txt"
    else:
        anno_name = img_path_list[cnt-1].split(".")[0] + ".txt"
    anno_path = anno_dir + "/" + anno_name
    return image,img_path_list[cnt],anno_path

def get_obj_anno(anno_path,box):
    WIDTH = 640
    HEIGHT = 471
    fr = open(anno_path)
    for line in fr.readlines():
        line = line.strip()
        strs = line.split(" ")
        center_x = float(strs[1]) * WIDTH
        center_y = float(strs[2]) * HEIGHT
        if center_x > box[0][0] and center_x < box[1][0] and center_y > box[0][1] and center_y < box[1][1]:    
            w = float(strs[3]) * WIDTH
            h = float(strs[4]) * HEIGHT
            if box[1][0] - box[0][0] < w:
                print("your annotation is small!")
            else:
                mx = (center_x - box[0][0])
                my = (center_y - box[0][1])
                mw = w
                mh = h
                new_anno_path = anno_save_dir + "/" + anno_path.split("/")[-1]
                fw = open(new_anno_path, "w")
                fw.write("0 ")
                fw.write(str(mx)+" "+str(my)+" "+str(mw)+" "+str(mh))
                fw.close()
                fr.close()
                return
    print("obj not found!")
    fr.close()
    return    

oriImage,_,__ = next_img()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cnt, oriImage

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            # cv2.imshow("Cropped", roi)
            cnt += 1
            oriImage, imgname, anno_path = next_img()
            cv2.imwrite(img_save_dir+"/"+imgname,roi)
            get_obj_anno(anno_path,refPoint)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while True:

    i = oriImage.copy()

    if not cropping:
        cv2.imshow("image", oriImage)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

    cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()