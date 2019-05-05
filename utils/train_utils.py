# encoding:utf-8
from backbones.resnet import resnet18, resnet50
from FCN import FCN32, FCN16, FCN8

from torchvision import models

def init_model(config_map, backbone_type_list=['resnet18', 'resnet50'], FCN_type_list=['FCN32s', 'FCN16s', 'FCN8s']):
    assert config_map['backbone'] in backbone_type_list, 'backbone not supported!!!'
    assert config_map['FCN_type'] in FCN_type_list, 'backbone not supported!!!'
    if config_map['backbone'] == backbone_type_list[0]:
        backbone_net = resnet18(input_size=config_map['image_size'])
        resnet = models.resnet18(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = backbone_net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        backbone_net.load_state_dict(dd)

    if config_map['backbone'] == backbone_type_list[1]:
        backbone_net = resnet50(input_size=config_map['image_size'])
        resnet = models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = backbone_net.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        backbone_net.load_state_dict(dd)
    
    if config_map['FCN_type'] == FCN_type_list[0]:
        fcn = FCN32(backbone_net, in_channel=config_map['in_channel'])
    
    if config_map['FCN_type'] == FCN_type_list[1]:
        fcn = FCN16(backbone_net, in_channel=config_map['in_channel'])

    if config_map['FCN_type'] == FCN_type_list[2]:
        fcn = FCN8(backbone_net, in_channel=config_map['in_channel'])

    return fcn

def init_lr(config_map):
    learning_rate = 0.0
    if config_map['resume_epoch'] > 0:
        for k, v in config_map['lr_adjust_map'].items():
            if k <= config_map['resume_epoch']:
                learning_rate = v 
    return learning_rate

def warmming_up_policy(now_iter, now_lr, stop_down_iter=1000):
    if now_iter <= stop_down_iter:
        now_lr += 0.000001
    return now_lr

def learning_rate_policy(now_iter, now_epoch, now_lr, lr_adjust_map, stop_down_iter=1000):
    now_lr = warmming_up_policy(now_iter, now_lr, stop_down_iter)
    if now_iter >= stop_down_iter and now_epoch in lr_adjust_map.keys():
        now_lr = lr_adjust_map[now_epoch]

    return now_lr
