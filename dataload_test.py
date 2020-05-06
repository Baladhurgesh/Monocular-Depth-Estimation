import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from model import DispNet_sequential as DispNet
from loss import SSIM,reconstruct_using_disparity, disparity_smoothness

import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian
# import Image
from PIL import Image



class KITTIDataset(Dataset):
    def __init__(self):
        self.rootdir = "/home/bala/DeepLearning/DL_project_test"
        self.left_images = []
        self.right_images = []
        self.height = 512
        self.length = 256
        self.lefimages = 0
        self.rigimages = 0
        for subdir, dirs, files in os.walk(self.rootdir):
            if "image_02/data" in subdir: #Left RGB Folder
                for file in files:
                    if ".jpg" in file:
                        # if self.lefimages>50:
                        #     break

                        left_file = os.path.join(subdir, file)
                        self.left_images.append(left_file)
                        self.lefimages+=1
                
            if "image_03/data" in subdir: #Right RGB Folder
                for file in files:
                    if ".jpg" in file:
                        # if self.rigimages>50:
                        #     break
                        right_file = os.path.join(subdir, file)
                        self.right_images.append(right_file)
                        self.rigimages+=1
        # print(len(self.left_images))
        self.left_images.sort()
        self.right_images.sort()
        # print(self.left_images)
        assert(len(self.left_images)==len(self.right_images))
       

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = cv2.imread(self.left_images[idx])
        right_img = cv2.imread(self.right_images[idx])

        PIL_left = Image.open(self.left_images[idx])
        PIL_right = Image.open(self.right_images[idx])

        np.asarray(left_img)
        np.asarray(right_img)
        
        left_img = cv2.resize(left_img,(self.height, self.length))
        right_img = cv2.resize(right_img,(self.height, self.length))

        left_img = np.moveaxis(left_img, 2,0)
        right_img = np.moveaxis(right_img, 2,0)

        return {"left_img":left_img,"right_img":right_img}

def scale_pyramid_(img, num_scales):
    # img = torch.mean(img, 1)
    # img = torch.unsqueeze(img, 1)
    scaled_imgs = [img]
    s = img.size()
    h = int(s[2])
    w = int(s[3])
    for i in range(num_scales-1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        temp = nn.functional.upsample(img, [nh, nw], mode='nearest')
        scaled_imgs.append(temp)
    return scaled_imgs

if __name__ == '__main__':
    dataset = KITTIDataset()
    # print(len(dataset))
    TrainLoader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, num_workers = 8)
    net = DispNet()
    net.to(torch.device("cuda:0"))
    net.load_state_dict(torch.load('/home/bala/DeepLearning/DL_Project/wts_monodepth.pth'))
    i = 0
    for sample_batched in TrainLoader:
        # print("training sample for KITTI")
        
        # print(sample_batched["left_img"].shape)
        net.zero_grad() 
        # print(sample_batched["right_img"].shape)
        left_original = sample_batched["left_img"]
        right_original = sample_batched["right_img"]


        left = left_original.type(torch.FloatTensor).cuda()
        right = right_original.type(torch.FloatTensor).cuda()
        # plt.figure(2)
        # plt.imshow(transforms.ToPILImage()(sample_batched["left_img"][0]))
        # plt.show()
        # plt.pause(0.25)
        left_pyramid = scale_pyramid_(left,4)
        right_pyramid = scale_pyramid_(right,4)

        output = net.forward(left)


        left_disp = [output[i][:, 0, :, :] for i in range(4)]
        right_disp = [output[i][:, 1, :, :] for i in range(4)]
        # print(left_disp.shape)
        i+=1
        rgb = right_disp[0][0].detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Horizontally stacked subplots')
        ax1.imshow(rgb,cmap='plasma')
        ax2.imshow(transforms.ToPILImage()(sample_batched["left_img"][0]))
        # fig = plt.figure(1)
        # plt.imshow(rgb,cmap='plasma')
        plt.savefig('/home/bala/DeepLearning/DL_project_test/epoch'+str(i+1)) 
        # plt.figure(2)
        # plt.imshow(transforms.ToPILImage()(sample_batched["left_img"][0]))
        # plt.show()
        # print("left image : ",left.shape)
        # print("right_disp : ",right_disp.shape)
        
        # print(left_reconstuct[0].shape)
        # recons = np.uint8(np.mean(left_reconstuct[0][0].detach().cpu().numpy(),axis = 0))
        # print(recons.shape)
        # plt.figure(3)
        # plt.imshow(transforms.ToPILImage()(recons))
        # plt.pause(0.5)
   