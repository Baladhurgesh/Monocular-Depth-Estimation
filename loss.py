import torch
import torch.nn as nn
from bilinear_sampler import apply_disparity

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.functional.avg_pool2d(x, 3, 1, padding = 0)
    mu_y = nn.functional.avg_pool2d(y, 3, 1, padding = 0)

    sigma_x  = nn.functional.avg_pool2d(x ** 2, 3, 1, padding = 0) - mu_x ** 2
    sigma_y  = nn.functional.avg_pool2d(y ** 2, 3, 1, padding = 0) - mu_y ** 2

    sigma_xy = nn.functional.avg_pool2d(x * y , 3, 1, padding = 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def reconstruct_using_disparity(image, disparity):
    return apply_disparity(image, disparity)

def gradient_x(image):
    return image[:,:,:,:-1]-image[:,:,:,1:]

def gradient_y(image):
    return image[:,:,:-1,:]-image[:,:,1:,:]

def disparity_smoothness(image, disparity):
    grad_img_x = [gradient_x(i) for i in image]
    grad_img_y = [gradient_y(i) for i in image]

    grad_disp_x = [gradient_x(i) for i in disparity]
    grad_disp_y = [gradient_y(i) for i in disparity]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in grad_img_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in grad_img_y]

    # weights_x = [-torch.mean(torch.abs(g), 1, keepdim=True) for g in grad_img_x]
    # weights_y = [-torch.mean(torch.abs(g), 1, keepdim=True) for g in grad_img_y]

    # smoothness_x = [torch.mean(grad_disp_x[i] * weights_x[i]) for i in range(4)]
    # smoothness_y = [torch.mean(grad_disp_y[i] * weights_y[i]) for i in range(4)]

    smoothness_x = [grad_disp_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [grad_disp_y[i] * weights_y[i] for i in range(4)]
    # dim = 1 8,1,256,511
    smoothness_x = [torch.nn.functional.pad(k,(0,1,0,0,0,0,0,0),mode='constant') for k in smoothness_x]
    smoothness_y = [torch.nn.functional.pad(k,(0,0,0,1,0,0,0,0),mode='constant') for k in smoothness_y]

    disp_smoothness = smoothness_x+smoothness_y

    disp_loss = [torch.mean(torch.abs(disp_smoothness[i])) / 2 ** i for i in range(4)]
        # self.disp_right_loss = [torch.mean(torch.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        # self.disp_gradient_loss = sum(self.disp_left_loss + self.disp_right_loss)

    # for i in range(4):
    #     print("disparity_smoothness weights : ",disp_loss[i])

    return disp_loss