import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
trans = transforms.Compose([
                transforms.ToTensor(),
            ])


class Config(object):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    display = False
    iterations = 1
    width, height = 120, 120 
    emp_iterations = 100
    max_pertubation_mask = 200 # 25, 50, 100, 200
    content = 0 # range(0, 1, 0.1)
    res_folder = "attack_res"
    attack_dataset_folder = "/workspace/shaped_mask_attack/yolov3/VOC2007/attack_clean"
    # texture = Image.open("/workspace/shaped_mask_attack/texture.png")
    texture = Image.new("RGB",(416,416),"white")
    r, g, b = texture.split()
    texture = Image.merge("RGB", (r, g, b))
    texture_input = trans(texture) 
    texture_ori = torch.stack([texture_input]).cuda()
    texture_ori = F.interpolate(texture_ori, (height, width), mode='bilinear', align_corners=False).cuda() # 采用双线性插值将不同大小图片上/下采样到统一大小
            
    # attack_dataset_folder = "~/shaped_mask_attack/at_training"

