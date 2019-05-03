# encoding:utf-8
import os, cv2, logging, numpy as np, time, json, argparse
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from backbones.resnet import resnet18, resnet50
from utils.FCNDataLoader import FCNDataset
from utils.utils import *
import multiprocessing as mp
from torchsummary import summary
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from utils.visual import Visual
from FCN import FCN32

parser = argparse.ArgumentParser(
    description='FCN Training params')
parser.add_argument('--config', default='configs/mask_resnet_sgd_7x7.json')
args = parser.parse_args()

config_map = get_config_map(args.config)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = init_lr(config_map)

backbone_net = init_model(config_map).to(device)
FCN_net = FCN32(backbone_net, in_channel=config_map['in_channel'])
backbone_net_p = nn.DataParallel(FCN_net.to(device), device_ids=config_map['gpu_ids'])
if config_map['resume_from_path']:
    backbone_net_p.load_state_dict(torch.load(config_map['resume_from_path']))

summary(backbone_net_p, (3, 448, 448), batch_size=config_map['batch_size'])

optimizer = torch.optim.SGD(backbone_net_p.parameters(), lr=learning_rate, momentum=0.99) # , weight_decay=5e-4)

if not os.path.exists(config_map['base_save_path']):
    os.makedirs(config_map['base_save_path'])

logger = create_logger(config_map['base_save_path'], config_map['log_name'])

my_vis = Visual(config_map['base_save_path'], log_to_file=config_map['vis_log_path'])

# backbone_net_p.load_state_dict(torch.load('densenet_sgd_S7_yolo.pth'))

backbone_net_p.train()

transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

train_dataset = FCNDataset(list_file=config_map['train_txt_path'], train=True, transform = transform, device=device, little_train=False)
train_loader = DataLoader(train_dataset,batch_size=config_map['batch_size'], shuffle=True, num_workers=4)
test_dataset = FCNDataset(list_file=config_map['test_txt_path'], train=False,transform = transform, device=device, little_train=False, with_file_path=True)
test_loader = DataLoader(test_dataset,batch_size=config_map['batch_size'],shuffle=False)#, num_workers=4)
data_len = int(len(test_dataset) / config_map['batch_size'])
logger.info('the dataset has %d images' % (len(train_dataset)))
logger.info('the batch_size is %d' % (config_map['batch_size']))

criterion = nn.BCEWithLogitsLoss()

num_iter = 0
best_mAP = 0.0
train_len = len(train_dataset) 
train_iter = config_map['resume_epoch'] * len(train_loader)
last_little_mAP = 0.0

my_vis.img('label colors', get_class_color_img())

for epoch in range(config_map['resume_epoch'], config_map['epoch_num']):
    backbone_net_p.train()

    logger.info('\n\nStarting epoch %d / %d' % (epoch + 1, config_map['epoch_num']))
    logger.info('Learning Rate for this epoch: {}'.format(optimizer.param_groups[0]['lr']))

    epoch_start_time = time.clock()
    
    total_loss = 0.
    avg_loss = 0.
    
    for i,(images, mask_label) in enumerate(train_loader):
        # print('mask label : ', mask_label.shape, mask_label.dtype)
        it_st_time = time.clock()
        train_iter += 1
        learning_rate = learning_rate_policy(train_iter, epoch, learning_rate, config_map['lr_adjust_map'], stop_down_iter=config_map['stop_down_iter'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        my_vis.plot('now learning rate', learning_rate)
        images = images.to(device)
        mask_label = mask_label.to(device)

        p_mask = backbone_net_p(images)
        # loss = cross_entropy2d(p_mask, mask_label)
        loss = criterion(p_mask, mask_label)
        total_loss += loss.data.item()

        if my_vis and i % config_map['show_img_iter_during_train'] == 0:
            backbone_net_p.eval()
            img = un_normal_trans(images[0])
            img = Tensor2Img(img)
            my_vis.img('origin img', img)
            my_vis.img('mask gt', mask_label_2_img(mask_label[0].byte().cpu().numpy()))
            my_vis.img('pred mask', mask_label_2_img(p_mask[0]))
            backbone_net_p.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        it_ed_time = time.clock()
        it_cost_time = it_ed_time - it_st_time
        if (i+1) % 5 == 0:
            avg_loss = total_loss / (i+1)
            logger.info('Epoch [%d/%d], Iter [%d/%d] expect end in %.2f min. Loss: %.4f, average_loss: %.4f, now learning rate: %f' %(epoch+1, config_map['epoch_num'], i+1, len(train_loader), it_cost_time * (len(train_loader) - i+1) // 60 , loss.item(), total_loss / (i+1), learning_rate))
            num_iter += 1
        
    epoch_end_time = time.clock()
    epoch_cost_time = epoch_end_time - epoch_start_time
    now_epoch_train_loss = total_loss / (i+1)
    my_vis.plot('train loss', now_epoch_train_loss)
    logger.info('Epoch {} / {} finished, cost time {:.2f} min. expect {} min finish train.'.format(epoch, config_map['epoch_num'], epoch_cost_time / 60, (epoch_cost_time / 60) * (config_map['epoch_num'] - epoch + 1)))


    torch.save(backbone_net_p.state_dict(),'%s/%s_last.pth'%(config_map['base_save_path'], config_map['backbone']))