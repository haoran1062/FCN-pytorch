# encoding:utf-8
import torch, numpy as np
import torch.nn as nn

from backbones.resnet import resnet18, resnet50
from torchsummary import summary


def bilinear_init(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

class FCN32(nn.Module):

    def __init__(self, backbone, in_channel=512, num_classes=21):
        super(FCN32, self).__init__()
        self.backbone = backbone
        self.cls_num = num_classes

        self.relu    = nn.ReLU(inplace=True)
        self.Conv1x1 = nn.Conv2d(in_channel, self.cls_num, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.cls_num)
        self.DCN32 = nn.ConvTranspose2d(self.cls_num, self.cls_num, kernel_size=64, stride=32, dilation=1, padding=16)
        self.DCN32.weight.data = bilinear_init(self.cls_num, self.cls_num, 64)
        self.dbn32 = nn.BatchNorm2d(self.cls_num)
    

    def forward(self, x):
        x0, x1, x2, x3, x4, x5, x6 = self.backbone(x)
        x = self.bn1(self.relu(self.Conv1x1(x5)))
        x = self.dbn32(self.relu(self.DCN32(x)))
        return x 

if __name__ == "__main__":
    from torchvision import transforms, utils
    device = 'cuda:0'
    backbone = resnet18()
    model = FCN32(backbone, in_channel=512)
    summary(model.to(device), (3, 448, 448))


    
    tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    in_img = np.zeros((448, 448, 3), np.uint8)
    t_img = transforms.ToTensor()(in_img)
    t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_img)
    t_img.unsqueeze_(0)
    t_img = t_img.to(device)
    
    x = model.forward(t_img)
    print(x.shape)


