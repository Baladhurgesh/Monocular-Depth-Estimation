import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

class DispNet_sequential(nn.Module) :
  def __init__(self):
    super(DispNet_sequential,self).__init__()
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    self.conv1 = nn.Conv2d(3,64,6,2,padding = (2,2))#256*128
    self.conv1_batch = nn.BatchNorm2d(64, affine = False)
    
    self.conv2 = nn.Conv2d(64,128,6,2,padding=(2,2))#128*64
    self.conv2_batch = nn.BatchNorm2d(128, affine = False)
    
    self.conv3a = nn.Conv2d(128,256,6,2,padding=(2,2))#64*32
    self.conv3a_batch = nn.BatchNorm2d(256, affine = False)
    
    self.conv3b = nn.Conv2d(256,256,3,1,padding=(1,1))#64*32
    self.conv3b_batch = nn.BatchNorm2d(256, affine = False)
    
    self.conv4a = nn.Conv2d(256,512,3,2,padding=(1,1))#32*16
    self.conv4a_batch = nn.BatchNorm2d(512, affine = False)
    
    self.conv4b = nn.Conv2d(512,512,3,1,padding=(1,1))#32*16
    self.conv4b_batch = nn.BatchNorm2d(512, affine = False)
    
    self.conv5a = nn.Conv2d(512,512,3,2,padding=(1,1))#16*8
    self.conv5a_batch =  nn.BatchNorm2d(512, affine = False)
    
    self.conv5b = nn.Conv2d(512,512,3,1,padding=(1,1))#16*8
    self.conv5b_batch = nn.BatchNorm2d(512, affine = False)
    
    self.conv6a = nn.Conv2d(512,1024,3,2,padding=(1,1))#8*4
    self.conv6a_batch = nn.BatchNorm2d(1024, affine = False)
    
    self.conv6b = nn.Conv2d(1024,1024,3,1,padding=(1,1)) #8*4
    self.conv6b_batch = nn.BatchNorm2d(1024, affine = False)
    
    self.upconv6b = nn.Upsample(scale_factor=2, mode='nearest') #16*8
    self.conv_upcon6b = nn.Conv2d(1024,1024,3,1,padding=(1,1))
    self.conv6_final = nn.Conv2d(1024+512,512,3,1,padding=(1,1))

    self.upconv5b = nn.Upsample(scale_factor=2, mode='nearest') #32*16
    self.conv_upcon5b = nn.Conv2d(512,512,3,1,padding=(1,1))
    self.conv5_final = nn.Conv2d(512+512,256,3,1,padding=(1,1))

    self.upconv4b = nn.Upsample(scale_factor=2, mode='nearest') #64*32
    self.conv_upcon4b = nn.Conv2d(256,256,3,1,padding=(1,1))
    self.conv4_final = nn.Conv2d(256+256,128,3,1,padding=(1,1))
    self.conv4_disp = nn.Conv2d(128, 2, 3, 1,padding=(1,1))
    self.conv4_dispup = nn.Upsample(scale_factor = 2, mode = 'nearest')

    self.upconv3b = nn.Upsample(scale_factor=2, mode='nearest') #128*64
    self.conv_upcon3b = nn.Conv2d(128, 128,3,1,padding=(1,1))
    self.conv3_final = nn.Conv2d(128+128+2,64,3,1,padding=(1,1))
    self.conv3_disp = nn.Conv2d(64, 2, 3, 1,padding=(1,1))
    self.conv3_dispup = nn.Upsample(scale_factor = 2, mode = 'nearest')

    self.upconv2 = nn.Upsample(scale_factor=2, mode='nearest') #256*128
    self.conv_upcon2 = nn.Conv2d(64,64,3,1,padding=(1,1))
    self.conv2_final = nn.Conv2d(64+64+2,32,3,1,padding=(1,1))
    self.conv2_disp = nn.Conv2d(32, 2, 3, 1,padding=(1,1))
    self.conv2_dispup = nn.Upsample(scale_factor = 2, mode = 'nearest')

    self.upconv1 = nn.Upsample(scale_factor=2, mode='nearest') #512*256
    self.conv_upcon1 = nn.Conv2d(32,32,3,1,padding=(1,1))
    self.conv1_final = nn.Conv2d(32+2,2,3,1,padding=(1,1))    
    

  def forward(self,x):

    x1 = F.relu(self.conv1_batch(self.conv1(x)))
    x2 = F.relu(self.conv2_batch(self.conv2(x1)))
    
    x3a = F.relu(self.conv3a_batch(self.conv3a(x2)))
    x3b = F.relu(self.conv3b_batch(self.conv3b(x3a)))
    
    x4a = F.relu(self.conv4a_batch(self.conv4a(x3b)))
    x4b = F.relu(self.conv4b_batch(self.conv4b(x4a)))
    
    x5a = F.relu(self.conv5a_batch(self.conv5a(x4b)))
    x5b = F.relu(self.conv5b_batch(self.conv5b(x5a)))
    
    x6a = F.relu(self.conv6a_batch(self.conv6a(x5b)))
    x6b = F.relu(self.conv6b_batch(self.conv6b(x6a)))
    
    x6up = F.relu(self.conv_upcon6b(self.upconv6b(x6b)))
    x5_final = F.relu(self.conv6_final(torch.cat((x6up,x5b),1)))

    x5up = F.relu(self.conv_upcon5b(self.upconv5b(x5_final)))
    x4_final = F.relu(self.conv5_final(torch.cat((x5up,x4b),1)))

    x4up = F.relu(self.conv_upcon4b(self.upconv4b(x4_final)))
    x3_final = F.relu(self.conv4_final(torch.cat((x4up,x3b),1)))
    x3_disp = F.relu(self.conv4_disp(x3_final))
    x3_dispup = self.conv4_dispup(x3_disp)

    x3up = F.relu(self.conv_upcon3b(self.upconv3b(x3_final)))
    x2_final = F.relu(self.conv3_final(torch.cat((x3up,x2,x3_dispup),1)))
    x2_disp = F.relu(self.conv3_disp(x2_final))
    x2_dispup = self.conv3_dispup(x2_disp)

    x2up = F.relu(self.conv_upcon2(self.upconv2(x2_final)))
    x1_final = F.relu(self.conv2_final(torch.cat((x2up,x1,x2_dispup),1)))
    x1_disp = F.relu(self.conv2_disp(x1_final))
    x1_dispup = self.conv2_dispup(x1_disp)

    x1up = F.relu(self.conv_upcon1(self.upconv1(x1_final)))
    x0_disp = F.relu(self.conv1_final(torch.cat((x1up,x1_dispup),1)))
    #TO DO D = 1/(aσ + b) Output Depth will be between 0.1 and 100
    return [x0_disp, x1_disp, x2_disp, x3_disp]
    
 