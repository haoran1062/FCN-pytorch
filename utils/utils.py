# encoding:utf-8
import os, numpy as np, random, cv2, logging, json
import torch

from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
palette=[]
for i in range(256):
    palette.extend((i,i,i))
palette[:3*21]=np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128]], dtype='uint8').flatten()
def PIL2cv(image):
    image.save('temp.png')
    return cv2.imread('temp.png')
    # return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def mask_label_2_img(mask_label):
    if isinstance(mask_label, np.ndarray):
        mask_label = torch.from_numpy(mask_label)
    _, index_mask = torch.max(mask_label, 0)
    np_mask = index_mask.byte().cpu().numpy()
    im=Image.fromarray(np_mask)
    im.putpalette(palette)
    return PIL2cv(im)
    
def Tensor2Img(in_tensor):
    if len(list(in_tensor.shape)) == 4:
        in_tensor = in_tensor[0]

    in_tensor = in_tensor.mul(255).byte().permute((1, 2, 0))
    img = in_tensor.cpu().numpy()
    # print(img)
    # print(img.shape)
    return img

VOC_CLASSES = (   
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]

def bbox_un_norm(bboxes, img_size=(448, 448)):
    (w, h) = img_size
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * w)
        bbox[1] = int(bbox[1] * h)
        bbox[2] = int(bbox[2] * w)
        bbox[3] = int(bbox[3] * h)
    return bboxes

def prep_test_data(file_path, little_test=None):
    target =  defaultdict(list)
    
    image_list = [] #image path list
    f = open(file_path)
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
        

    f.close()

    if little_test:
        file_list = file_list[:little_test]

    print('---prepare target---')
    img_size = (448, 448)
    bar = tqdm(total=len(file_list))
    for index,image_file in enumerate(file_list):

        image_id = image_file.split('/')[-1].split('.')[0]
        image_list.append(image_id)
        label_list = from_img_path_get_label_list(image_file, img_size=img_size)
        for i in label_list:
            
            class_name = VOC_CLASSES[i[0]]
            target[(image_id,class_name)].append(i[1:])
       
        bar.update(1)
    bar.close()
    return target

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def cv_resize(img, resize=448):
    return cv2.resize(img, (resize, resize))

def create_logger(base_path, log_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger


def get_config_map(file_path):
    config_map = json.loads(open(file_path).read())
    temp_map = {}
    for k, v in config_map['lr_adjust_map'].items():
        temp_map[int(k)] = v
    config_map['lr_adjust_map'] = temp_map
    config_map['batch_size'] *= len(config_map['gpu_ids'])
    return config_map



def addImage(img, img1): 
    
    h, w, _ = img1.shape 
    # 函数要求两张图必须是同一个size 
    img2 = cv2.resize(img, (w,h), interpolation=cv2.INTER_AREA) #print img1.shape, img2.shape #alpha，beta，gamma可调 
    alpha = 0.5
    beta = 1-alpha 
    gamma = 0 
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add

def get_class_color_img():
    img = np.zeros((750, 300, 3), np.uint8)
    h, w, c = img.shape
    img.fill(255)
    color_img = np.zeros(img.shape, np.uint8)
    clsn = 20
    cross = int(h / clsn)
    for i in range(clsn):
        color_img[i*cross:(i+1)*cross] =  np.array(Color[i], np.uint8)
        cv2.putText(img, '%s'%(VOC_CLASSES[i]), (30, int(i * cross) + int(cross/1.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, 30)
    img = addImage(img, color_img)
    return img

