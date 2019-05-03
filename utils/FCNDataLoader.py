# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
from utils import *
ia.seed(random.randint(1, 10000))


class FCNDataset(data.Dataset):
    image_size = 448
    def __init__(self,list_file,train,transform, device, little_train=False, with_file_path=False, B = 2, C = 21, test_mode=False):
        print('data init')
        
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.resize = 448
        self.B = B
        self.C = C
        self.device = device
        self._test = test_mode
        self.with_file_path = with_file_path
        self.img_augsometimes = lambda aug: iaa.Sometimes(0.25, aug)

        self.augmentation = iaa.Sequential(
            [
                # augment without change bboxes 
                self.img_augsometimes(
                    iaa.SomeOf((1, 3), [
                        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
                        iaa.Sharpen((0.1, .8)),       # sharpen the image
                        # iaa.GaussianBlur(sigma=(2., 3.5)),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(2., 3.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.BilateralBlur(d=(7, 12), sigma_color=(10, 250), sigma_space=(10, 250)),
                            iaa.MedianBlur(k=(3, 7)),
                        ]),
                        

                        iaa.AddElementwise((-50, 50)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                        iaa.JpegCompression(compression=(80, 95)),

                        iaa.Multiply((0.5, 1.5)),
                        iaa.MultiplyElementwise((0.5, 1.5)),
                        iaa.ReplaceElementwise(0.05, [0, 255]),
                        # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                        #                 children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        iaa.OneOf([
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(1, iaa.Add((-10, 50)))),
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        ]),

                    ], random_order=True)
                ),

            ],
            random_order=True
        )

        # torch.manual_seed(23)
        with open(list_file) as f:
            lines  = f.readlines()
        
        if little_train:
            lines = lines[:64*8]

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            
        self.num_samples = len(self.fnames)
    
    def get_bit_mask(self, mask_img, cls_n=21):
        '''
            mask_img : cv2 numpy mat BGR
            return [NxHxW] numpy
        '''
        b = mask_img
        
        h, w = b.shape
        bit_mask = np.zeros((cls_n, h, w), np.uint8)
        for i in range(cls_n):
            bit_mask[i, :, :] = np.where(b==i, 1, 0)

        return bit_mask
    
    def load_mask_label(self, file_path, resize=448):
        """
            Load label image as 1 x height x width integer array of label indices.
            The leading singleton dimension is required by the loss.
        """
        im = Image.open(file_path.replace('JPEGImages', 'SegmentationClass').replace('jpg', 'png'))
        im = im.resize((resize, resize))
        label = np.array(im, dtype=np.uint8)
        label = np.where(label==255, 0, label)
        # label = label[np.newaxis, ...]
        return torch.from_numpy(self.get_bit_mask(label)).float() #.to(self.device)# .byte()

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        if self._test:
            print(fname)
        img = cv2.imread(fname)
        mask_label = self.load_mask_label(fname)
        


        if self.train:
            # TODO
            # add data augument
            # print('before: ')
            # print(boxes)
            
            seq_det = self.augmentation.to_deterministic()
            img = seq_det.augment_images([img])[0]
            
            # pass
                
        img = self.transform(img)
        # print(fname)
        if self.with_file_path:
            return img, mask_label, fname
        
        return img, mask_label
        # return img.to(self.device), target.to(self.device)

    def __len__(self):
        return self.num_samples


    
if __name__ == "__main__":

    from utils import cv_resize, Color

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    S = 7
    train_dataset = FCNDataset(list_file='datasets/2012_seg.txt',train=False,transform = transform, test_mode=True, device='cuda:0')
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    # print(next(train_iter))
    for i in range(200):
        print('~'*50 + '\n\n\n')
        img, mask_label = next(train_iter)
        # mask_img = mask_img.squeeze(0).cpu().numpy()
        # print('mask shape is :', mask_img.shape)
        # print(img.shape, target.shape)
        # print(boxes, clss, confs)
        

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img = un_normal_trans(img.squeeze(0))
        img = Tensor2Img(img)
        cv2.imshow('img', img)
        # print(mask_label[0, 10:100, 10:100])
        
        mask_img = mask_label_2_img(mask_label[0])
        cv2.imshow('mask', mask_img)
        if cv2.waitKey(12000)&0xFF == ord('q'):
            break

    # for i in range(7):
    #     for j in range(7):
    #         print(target[:, i:i+1, j:j+1, :])
