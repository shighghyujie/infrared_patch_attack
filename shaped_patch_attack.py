import os
import joblib
import time
import argparse
from config import Config
from torchvision import datasets, transforms
from attack.mask_attack import shaped_mask_attack
from yolov3.detect import load_model, detect
# from yolov7.detect_attack import load_model, detect
# from yolov5.detect_attack import load_model, detect

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

loader = transforms.Compose([
    transforms.ToTensor()
])

def attack_process(H, W, img, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name, lambda_sparse, lambda_attack, lambda_agg):
    input = loader(img)
    bbox, prob = detect(threat_model, input) # 在攻击前检测原目标的置信度
    print("obj score before attack: ", prob.item())
    if prob.item()<0.5: # 本身检测分数较低的跳过
        return 
    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack(H, W, bbox, threat_model, img, device, emp_iterations, max_pertubation_mask, content, lambda_sparse, lambda_attack, lambda_agg) # 调用攻击函数进行攻击
    _, prob = detect(threat_model,adv_img_ts) # 在攻击后检测目标的置信度
    end = time.time()
    print("optimization time: {}".format(end - begin))
    res.append(prob.item())
    print("obj score after attack3: ",prob.item())
    inter = "sample" + str(args.number) + "+" + str(args.max_pertubation_mask)
    if prob.item() <= min(res):
        file_path = folder_path + '/res/' + inter + '/{}.jpg'.format(name)
        adv_img.save(file_path,quality=99)
        joblib.dump(adv_img_ts,folder_path+"/res/adv_ts" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_pedestrian&params.pkl".format(name))
        joblib.dump(mask,folder_path+"/res/mask" + str(args.number) + "+" + str(args.max_pertubation_mask) + "/{}_mask&params.pkl".format(name))  
    return


if __name__=="__main__":
    
    ## 加载攻击参数 ##
    opt = Config()
    folder_path = opt.res_folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--content',type=float,default=0, help='the content of the mask')
    parser.add_argument('--number',type=int,default=777, help='the number of the train')
    parser.add_argument('--max_pertubation_mask',type=int,default=200, help='the size of mask')
    parser.add_argument('--lambda_sparse', type=float, default=5, help='the para of L_sparse')
    parser.add_argument('--lambda_attack', type=float, default=20, help='the size of L_attack')
    parser.add_argument('--lambda_agg', type=float, default=25, help='the size of L_agg')

    args = parser.parse_args()

    ## 保存攻击后的结果 ##
    if not os.path.exists(folder_path + "/res/sample" + str(args.number)+"+"+ str(args.max_pertubation_mask)): 
        os.makedirs(folder_path + "/res/sample" + str(args.number) + "+" +str(args.max_pertubation_mask))
    if not os.path.exists(folder_path +"/res/adv_ts" + str(args.number)+"+"+ str(args.max_pertubation_mask)): 
        os.makedirs(folder_path + "/res/adv_ts" + str(args.number) + "+" +  str(args.max_pertubation_mask))
    if not os.path.exists(folder_path +"/res/mask" + str(args.number)+ "+" + str(args.max_pertubation_mask)): 
        os.makedirs(folder_path + "/res/mask" + str(args.number) +"+" + str(args.max_pertubation_mask))
    
    ## 创建目标数据集 ##
    dataset = datasets.ImageFolder(opt.attack_dataset_folder)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}


    opt.iterations = 3
    ## 加载待攻击模型 ##
    threat_model = load_model()
    threat_model.eval()
    for i in range(len(dataset)):
        if i == 10:
            print()
        res = []

        for k in range(opt.iterations):
            idx = i
            img = dataset[idx][0]
            label = dataset[idx][1]
            name = dataset.samples[idx][0].split(".")[0].split("/")[-1]
            print("{}th image attack".format(i))
            attack_process(opt.height, opt.width, img, threat_model, opt.device, opt.emp_iterations, args.max_pertubation_mask, args.content, folder_path, name, args.lambda_sparse, args.lambda_attack, args.lambda_agg)

           