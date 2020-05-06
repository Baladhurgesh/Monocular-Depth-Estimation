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
        self.rootdir = "/home/bala/DeepLearning/DL_Project"
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
    TrainLoader = torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True, num_workers = 8)
    net = DispNet()
    net.to(torch.device("cuda:0"))
    loss_function = nn.L1Loss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    alpha = 0.3
    for epoch in range(50):
        for sample_batched in TrainLoader:
            # print("training sample for KITTI")
            
            # print(sample_batched["left_img"].shape)
            net.zero_grad() 
            # print(sample_batched["right_img"].shape)
            left_original = sample_batched["left_img"]
            right_original = sample_batched["right_img"]

            # pyramid = tuple(pyramid_gaussian(left_original,  max_layer=4, downscale=2, multichannel=True))
            # # print("pyramid",size(pyramid))
            
            # pyr_batch = [pyramid_gaussian(item,  max_layer=4, downscale=2, multichannel=True) for item in left_original]     
            
            # #pyr_batch_axis_moved = [np.moveaxis(p, 2,0) for p in pyr_batch]
            # for p in pyr_batch:
            #     print("level: ",p.shape)
            # print("left_original",left_original.shape)
            # print("right_original",right_original.shape)

            left = left_original.type(torch.FloatTensor).cuda()
            right = right_original.type(torch.FloatTensor).cuda()

            left_pyramid = scale_pyramid_(left,4)
            right_pyramid = scale_pyramid_(right,4)

            # for p in left_pyramid:
            #     print("pyr =" ,p.shape)
            output = net.forward(left)

            #[8,3, 256, 512]
            # print("output :",output.shape)
            left_disp = [output[i][:, 0, :, :] for i in range(4)]
            right_disp = [output[i][:, 1, :, :] for i in range(4)]
            # print(left_disp.shape)
            
            right_reconstuct = [reconstruct_using_disparity(left_pyramid[i], right_disp[i]) for i in range(4)]
            left_reconstuct = [reconstruct_using_disparity(right_pyramid[i], left_disp[i]) for i in range(4)]
            
            # TODO: Put weighted loss for pyramid : error in smaller image should contribute more
            left_L1loss = [loss_function(left_pyramid[i], left_reconstuct[i]) for i in range(4)]
            right_L1loss = [loss_function(right_pyramid[i], right_reconstuct[i]) for i in range(4)]
            L1_loss = torch.FloatTensor([0]).cuda()
            SSIM_loss = torch.FloatTensor([0]).cuda()
            LR_loss_fin = torch.FloatTensor([0]).cuda()

            for i in range(4): 
                L1_loss += (left_L1loss[i] + right_L1loss[i])
            L1_loss/=4 
            # print("L1 loss: ", L1_loss)
            left_SSIM_loss = [torch.mean(SSIM(left_pyramid[i], left_reconstuct[i])) for i in range(4)] #Reconstructed Image and Original Image 
            right_SSIM_loss = [torch.mean(SSIM(right_pyramid[i], right_reconstuct[i])) for i in range(4)]
            for i in range(4): 
                SSIM_loss += (left_SSIM_loss[i] + right_SSIM_loss[i])
            SSIM_loss/=4
            # print("SSIM LOSS: ", SSIM_loss)
            # left_SSIM_loss = left_SSIM_loss[0] + left_SSIM_loss[1] + left_SSIM_loss[2] + left_SSIM_loss[3]
            app_match_loss = (alpha * (1 - SSIM_loss)/2) + (1- alpha)*L1_loss

            # left_app_match_loss = (alpha * (1 - left_SSIM_loss)/2) + (1- alpha)*left_L1loss
            # right_app_match_loss = (alpha * (1 - right_SSIM_loss)/2) + (1- alpha)*right_L1loss

            # print("app loss: ", app_match_loss)

            left_disp[0] = left_disp[0].view([-1, 1, 256, 512])
            left_disp[1] = left_disp[1].view([-1, 1, 128, 256])
            left_disp[2] = left_disp[2].view([-1, 1, 64, 128])
            left_disp[3] = left_disp[3].view([-1, 1, 32, 64])


            # print(left_disp.shape)

            listt = [reconstruct_using_disparity(left_disp[i], right_disp[i]) for i in range(4)]    
            LR_loss = [torch.mean(left_disp[i]-listt[i]) for i in range(4)]
            for i in range(4): 
                LR_loss_fin += LR_loss[i] 
            LR_loss_fin/=4

            right_disp[0] = right_disp[0].view([-1, 1, 256, 512])
            right_disp[1] = right_disp[1].view([-1, 1, 128, 256])
            right_disp[2] = right_disp[2].view([-1, 1, 64, 128])
            right_disp[3] = right_disp[3].view([-1, 1, 32, 64])

            DSPsmooth_left_loss = disparity_smoothness(left_pyramid,left_disp)
            DSPsmooth_right_loss = disparity_smoothness(right_pyramid,right_disp)

            DSPsmooth_loss = sum(DSPsmooth_left_loss+DSPsmooth_right_loss)

            # print("disparity_smoothness :", DSPsmooth_loss)

            # print("LR_loss: ", LR_loss)
            # left_disp_level1 = output[1][:, 0, :, :]
            # right_disp_level1 = output[1][:, 1, :, :]
            # left_level1 = cv2.pyrDown(left_original)
            # print(left)
            
            # right_reconstuct = reconstruct_using_disparity(left, right_disp)
            # left_reconstuct = reconstruct_using_disparity(right, left_disp)

            # left_L1loss = loss_function(left, left_reconstuct)
            # right_L1loss = loss_function(right, right_reconstuct)

            # left_SSIM_loss = SSIM(left, left_reconstuct)#Reconstructed Image and Original Image 
            # right_SSIM_loss = SSIM(right, right_reconstuct)
            
            # left_app_match_loss = alpha * (1 - left_SSIM_loss)/2 + (1- alpha)*left_L1loss
            # right_app_match_loss = alpha * (1 - right_SSIM_loss)/2 + (1- alpha)*right_L1loss

            loss = app_match_loss/8 + 0.0*LR_loss_fin/8 + 0.1 * DSPsmooth_loss/8           
            loss.backward() 
            optimizer.step()
        #TO DO: Query same image and see how it evolves over epochs
        print("Epoch : ", epoch)
        print(loss)
        rgb = right_disp[0][0].detach().cpu().numpy()
        fig = plt.figure(1)
        plt.imshow(rgb,cmap='plasma')
        plt.savefig('/home/bala/DeepLearning/DL_Project/epoch/'+str(epoch)) 
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
        torch.save(net.state_dict(),'/home/bala/DeepLearning/DL_Project/wts_monodepth_3.pth')