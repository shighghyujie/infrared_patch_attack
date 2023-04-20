
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov3.models.common import DetectMultiBackend
from yolov3.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov3.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov3.utils.plots import Annotator, colors, save_one_box
from yolov3.utils.torch_utils import select_device, time_sync
from yolov3.utils.augmentations import letterbox
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms

device = torch.device("cuda")
# inputsize = [640,544]
inputsize = [416,416]
trans = transforms.Compose([
    transforms.ToTensor(),
])
# inputsize1 = [120,120]
# inputsize2 = [640,544]

def load_model():
    weights = "yolov3/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_model_infrared():
    weights = "/workspace/shaped_mask_cross_attack/weights/infrared_model.pt"
    # weights = "/workspace/shaped_mask_cross_attack/yolov3/runs/train/exp10/weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def load_model_rgb():
    weights = "/workspace/shaped_mask_cross_attack/weights/visible_model_hy.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False)
    return model

def detect_train(model,img):
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    # img_ = nn.functional.interpolate(img, (50, 50), mode='bilinear', align_corners=False).cuda()
    img_ = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False).cuda()
    pred = model(img_)
    sorted, rank = torch.sort(pred[0,:,4],descending=True)
    # loss = torch.sum(pred[0][rank[:100]],dim=0)[4]
    loss = torch.sum(pred[0][rank[:10]],dim=0)[4]
    # pred = non_max_suppression(pred, 0.05, 0.45, None, False, max_det=1000)
    return -loss

def detect(model,img,H=120,W=120):
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
    img = img.cuda()
    pred = model(img)
    # conf_thres=0.01 # confidence threshold
    conf_thres=0.00001 # confidence threshold
    iou_thres=0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    if len(pred[0]) == 0:
        return None,0
    left = max(int(pred[0][0][0].item()),0)
    up = max(int(pred[0][0][1].item()),0)
    # right = int(pred[0][0][2].item())
    # below = int(pred[0][0][3].item())
    right = min(int(pred[0][0][2].item()),inputsize[0])
    below = min(int(pred[0][0][3].item()),inputsize[1])
    return (up, below, left, right), pred[0][0][4].clone().detach()



# def detect(model,img,H=120,W=120):
#     if len(img.shape) == 3:
#         img = img[None]  # expand for batch dim
#     # img = nn.functional.interpolate(img, (100, 100), mode='bilinear', align_corners=False)
#     img = nn.functional.interpolate(img, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)
#     img = img.cuda()
#     pred = model(img)
#     # conf_thres=0.01 # confidence threshold
#     conf_thres=0.00001 # confidence threshold
#     iou_thres=0.45
#     pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
#     if len(pred[0]) == 0:
#         return None,0
#     left = max(int(pred[0][0][0].item()),0)
#     up = max(int(pred[0][0][1].item()),0)
#     # right = int(pred[0][0][2].item())
#     # below = int(pred[0][0][3].item())
#     right = min(int(pred[0][0][2].item()),inputsize[0])
#     below = min(int(pred[0][0][3].item()),inputsize[1])
#     # left = int(left*1.25)
#     # up = int(up*1.25)
#     # right = int(right/1.25)
#     # below = int(below/1.25)
#     # left = int(left*W/inputsize[0])#+2
#     # up = int(up*H/inputsize[1])#+5
#     # right = int(right*W/inputsize[0])#-2
#     # below = int(below*H/inputsize[1])#-5
#     # objmask = torch.zeros((H, W)).to(device) 
#     # objmask[up:below, left:right] = torch.ones(((below-up,right-left)))
#     return (up, below, left, right), pred[0][0][4].clone().detach()

