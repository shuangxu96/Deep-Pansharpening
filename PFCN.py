# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:56:16 2020

@author: win10
"""

'''
Networks
'''
import torch.nn as nn
import torch.nn.functional as F

import torch

def TV(x):
    '''
    Modified from https://github.com/jxgu1016/Total_Variation_Loss.pytorch
    '''
    h_x = x.size()[2]
    w_x = x.size()[3]
    h_tv = torch.abs((x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.abs((x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv, w_tv

class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self,x,y):
        x_h, x_w = TV(x)
        y_h, y_w = TV(y)
        return self.l1_loss(x_h,y_h)+self.l1_loss(x_w,y_w)

    
class fusionLayer(nn.Module):
    def __init__(self, channel, out_channels):
        super(fusionLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, out_channels, 1)
    
    def forward(self, ms, ls):
        return self.conv1x1(ms)+ls

class PFCN8(nn.Module):
    def __init__(self, Channel, channel):
        super(PFCN8, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv2d(Channel, 64, 3, padding=1))
        self.encoder.add_module('relu1', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp1', nn.MaxPool2d(2,stride=2))
        self.encoder.add_module('conv2', nn.Conv2d(64, 128, 3, padding=1))
        self.encoder.add_module('relu2', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp12', nn.MaxPool2d(2,stride=2))
        self.encoder.add_module('conv3', nn.Conv2d(128, 128, 3, padding=1))
        self.encoder.add_module('relu3', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp13', nn.MaxPool2d(2,stride=2))
        
        self.fusion1 = fusionLayer(channel, 128)
        self.fusion2 = fusionLayer(channel, 128)
        self.fusion3 = fusionLayer(channel, 128)
        self.fusion4 = fusionLayer(channel, 64)
        
        self.upscale1 = nn.Sequential()
        self.upscale1.add_module('conv3', nn.Conv2d(128,256,3,padding=1))
        self.upscale1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.upscale1.add_module('deconv3', nn.ConvTranspose2d(256,128,2,stride=2))
        self.upscale1.add_module('relu3_2', nn.ReLU(inplace=True))
        
        self.upscale2 = nn.Sequential()
        self.upscale2.add_module('conv4', nn.Conv2d(128,128,3,padding=1))
        self.upscale2.add_module('relu4_1', nn.ReLU(inplace=True))
        self.upscale2.add_module('deconv4', nn.ConvTranspose2d(128,128,2,stride=2))
        self.upscale2.add_module('relu4_2', nn.ReLU(inplace=True))
        
        self.upscale3 = nn.Sequential()
        self.upscale3.add_module('conv5', nn.Conv2d(128,128,3,padding=1))
        self.upscale3.add_module('relu5_1', nn.ReLU(inplace=True))
        self.upscale3.add_module('deconv5', nn.ConvTranspose2d(128,64,2,stride=2))
        self.upscale3.add_module('relu5_2', nn.ReLU(inplace=True))
        
        self.recon = nn.Sequential()
        self.recon.add_module('conv6', nn.Conv2d(64,64,3,padding=1))
        self.recon.add_module('relu6', nn.ReLU(inplace=True))
        self.recon.add_module('conv7', nn.Conv2d(64,Channel, 3, padding=1))
        self.recon.add_module('relu7', nn.ReLU(inplace=True))
        
    def forward(self, ms, ls):
        ms_2 = F.interpolate(ms, scale_factor=0.5)
        ms_4 = F.interpolate(ms, scale_factor=0.25)
        ms_8 = F.interpolate(ms, scale_factor=0.125)
        
        ls = self.encoder(ls)
        ls = self.upscale1(self.fusion1(ms_8,ls))
        ls = self.upscale2(self.fusion2(ms_4,ls))
        ls = self.upscale3(self.fusion3(ms_2,ls))
        ls = self.recon(self.fusion4(ms,ls))
        return ls

class PFCN4(nn.Module):
    def __init__(self, Channel, channel):
        super(PFCN4, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv2d(Channel, 64, 3, padding=1))
        self.encoder.add_module('relu1', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp1', nn.MaxPool2d(2,stride=2))
        self.encoder.add_module('conv2', nn.Conv2d(64, 128, 3, padding=1))
        self.encoder.add_module('relu2', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp12', nn.MaxPool2d(2,stride=2))
        
        self.fusion1 = fusionLayer(channel, 128)
        self.fusion2 = fusionLayer(channel, 128)
        self.fusion3 = fusionLayer(channel, 64)
        
        self.upscale1 = nn.Sequential()
        self.upscale1.add_module('conv3', nn.Conv2d(128,256,3,padding=1))
        self.upscale1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.upscale1.add_module('deconv3', nn.ConvTranspose2d(256,128,2,stride=2))
        self.upscale1.add_module('relu3_2', nn.ReLU(inplace=True))
        
        self.upscale2 = nn.Sequential()
        self.upscale2.add_module('conv4', nn.Conv2d(128,128,3,padding=1))
        self.upscale2.add_module('relu4_1', nn.ReLU(inplace=True))
        self.upscale2.add_module('deconv4', nn.ConvTranspose2d(128,64,2,stride=2))
        self.upscale2.add_module('relu4_2', nn.ReLU(inplace=True))
        
        self.recon = nn.Sequential()
        self.recon.add_module('conv5', nn.Conv2d(64,64,3,padding=1))
        self.recon.add_module('relu5', nn.ReLU(inplace=True))
        self.recon.add_module('conv6', nn.Conv2d(64,Channel, 3, padding=1))
        self.recon.add_module('relu6', nn.ReLU(inplace=True))
        
    def forward(self, ms, ls):
        ms_2 = F.interpolate(ms, scale_factor=0.5)
        ms_4 = F.interpolate(ms, scale_factor=0.25)
        
        ls = self.encoder(ls)
        ls = self.upscale1(self.fusion1(ms_4,ls))
        ls = self.upscale2(self.fusion2(ms_2,ls))
        ls = self.recon(self.fusion3(ms,ls))
        return ls

class PFCN2(nn.Module):
    def __init__(self, Channel, channel):
        super(PFCN2, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('conv1', nn.Conv2d(Channel, 64, 3, padding=1))
        self.encoder.add_module('relu1', nn.ReLU(inplace=True))
        self.encoder.add_module('maxp1', nn.MaxPool2d(2,stride=2))
        
        self.fusion1 = fusionLayer(channel, 64)
        self.fusion2 = fusionLayer(channel, 64)
        
        self.upscale = nn.Sequential()
        self.upscale.add_module('conv2', nn.Conv2d(64,128,3,padding=1))
        self.upscale.add_module('relu2_1', nn.ReLU(inplace=True))
        self.upscale.add_module('deconv2', nn.ConvTranspose2d(128,64,2,stride=2))
        self.upscale.add_module('relu2_2', nn.ReLU(inplace=True))

        self.recon = nn.Sequential()
        self.recon.add_module('conv3', nn.Conv2d(64,64,3,padding=1))
        self.recon.add_module('relu3', nn.ReLU(inplace=True))
        self.recon.add_module('conv4', nn.Conv2d(64,Channel, 3, padding=1))
        self.recon.add_module('relu4', nn.ReLU(inplace=True))
        
    def forward(self, ms, ls):
        ms_2 = F.interpolate(ms, scale_factor=0.5)
        
        ls = self.encoder(ls)
        ls = self.upscale(self.fusion1(ms_2,ls))
        ls = self.recon(self.fusion2(ms,ls))
        return ls

'''
Training
'''
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime
import os
from utils import H5Dataset, MMSRDataset
from tensorboardX import SummaryWriter
from kornia.losses import psnr_loss, ssim

# data & loader
batch_size = 64
num_epoch = 100
learning_rate = 1e-3

data_name = 'cave'
scale     = 2
if data_name=='cave':
    n_Channel = [31,3]
elif data_name=='quickbird':
    n_Channel = [4,1]
trainloader      = DataLoader(H5Dataset(r'D:\py_code\CSC-Fusion-4\cave_train.h5'),      
                              batch_size=batch_size, shuffle=True) 
validationloader = DataLoader(MMSRDataset(r'D:\data\MMSR\scale2\validation',scale),      
                              batch_size=1)
loader = {'train':      trainloader,
          'validation': validationloader}

# network
if scale==2:
    net = PFCN2(*n_Channel).to('cuda')
if scale==4:
    net = PFCN4(*n_Channel).to('cuda')
if scale==8:
    net = PFCN8(*n_Channel).to('cuda')
optimizer    = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
lossfun = nn.MSELoss()
gdl = GDL()
weight_gdl = 1

# logs
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('PFCN_logs',timestamp+'_bs%d_epoch%d_lr%.5f_scale%d_%s'%(batch_size,num_epoch,learning_rate,scale,data_name))
writer = SummaryWriter(save_path)

step = 0
best_psnr_val,psnr_val = 0., 0.
best_ssim_val,ssim_val = 0., 0.
torch.backends.cudnn.benchmark = True
for epoch in range(num_epoch):
    ''' train '''
    for i, (lr, guide, gt) in enumerate(loader['train']):
        net.train()
        
        # 1. update
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        lr = F.interpolate(lr, size=guide.shape[-2:])
        optimizer.zero_grad()
        gt_hat       = net(guide,lr)
        loss         = lossfun(gt,gt_hat)+weight_gdl*gdl(gt,gt_hat)
        loss.backward()
        optimizer.step()
        
        #2.  print
        print("[%d,%d] Loss:%.4f, PSNR: %.4f, SSIM: %.4f" %
                (epoch+1, i+1, 
                 loss.item(),
                 psnr_val, ssim_val))
        #3. Log the scalar values
        writer.add_scalar('loss', loss.item(), step)
        step+=1
    
    ''' validation ''' 
    psnr_val = 0.
    ssim_val = 0.
    with torch.no_grad():
        net.eval()
        for i, (lr, guide, gt) in enumerate(loader['validation']):
            lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
            lr = F.interpolate(lr, size=guide.shape[-2:])
            imgf = torch.clamp(net(guide, lr), 0., 1.)
            psnr_val += psnr_loss(imgf, gt, 1.)
            ssim_val += ssim(imgf, gt, 5, 'mean', 1.)
        psnr_val = float(psnr_val/loader['validation'].__len__())
        ssim_val = 1-2*float(ssim_val/loader['validation'].__len__())
    writer.add_scalar('PSNR on validation data', psnr_val, epoch)
    writer.add_scalar('SSIM on validation data', ssim_val, epoch)

    
    ''' decay the learning rate '''
#    scheduler.step()
    
    ''' save model ''' 
    if best_psnr_val<psnr_val:
        best_psnr_val = psnr_val
        torch.save(net.state_dict(), os.path.join(save_path, 'best_net.pth'))
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch},
                os.path.join(save_path, 'last_net.pth'))
    
'''
Test
'''
from scipy.io import savemat
net.load_state_dict(torch.load(os.path.join(save_path,'best_net.pth')))
testloader = DataLoader(MMSRDataset(r'D:\data\MMSR\scale2\test', scale),      
                              batch_size=1)

metrics = torch.zeros(2,testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (lr, guide, gt) in enumerate(testloader):
        lr, guide, gt = lr.cuda(), guide.cuda(), gt.cuda()
        lr = F.interpolate(lr, size=guide.shape[-2:])
        imgf = torch.clamp(net(guide, lr), 0., 1.)
        metrics[0,i] = psnr_loss(imgf, gt, 1.)
        metrics[1,i] = 1-2*ssim(imgf, gt, 5, 'mean', 1.)
        savemat(os.path.join(save_path,testloader.dataset.files[i].split('\\')[-1]),
               {'HR':imgf.squeeze().detach().cpu().numpy()} )

import xlwt
f = xlwt.Workbook()
sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
img_name = [i.split('\\')[-1].replace('.mat','') for i in testloader.dataset.files]
metric_name = ['PSNR','SSIM']
for i in range(len(metric_name)):
    sheet1.write(i+1,0,metric_name[i])
for j in range(len(img_name)):
   sheet1.write(0,j+1,img_name[j])  # 顺序为x行x列写入第x个元素
for i in range(len(metric_name)):
    for j in range(len(img_name)):
        sheet1.write(i+1,j+1,float(metrics[i,j]))
sheet1.write(0,len(img_name)+1,'Mean')
for i in range(len(metric_name)):
    sheet1.write(i+1,len(img_name)+1,float(metrics.mean(1)[i]))
f.save(os.path.join(save_path,'test_result.xls'))
 
