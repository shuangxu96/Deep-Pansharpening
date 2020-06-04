# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:08:20 2020

@author: win10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TFN(nn.Module):
    def __init__(self, Channel, channel):
        super(TFN, self).__init__()
        
        self.leaky_relu = nn.LeakyReLU(1e-2,True)
        
        self.ms_cnn = nn.Sequential()
        self.ms_cnn.add_module('conv1', nn.Conv2d(Channel, 32, 3, padding=1))
        self.ms_cnn.add_module('relu1', nn.LeakyReLU(1e-2,True))
        self.ms_cnn.add_module('conv2', nn.Conv2d(32, 32, 3, padding=1))
        self.ms_cnn.add_module('relu2', nn.LeakyReLU(1e-2,True))
        self.ms_cnn_down = nn.Conv2d(32, 64, 2, stride=2)
        
        self.pan_cnn = nn.Sequential()
        self.pan_cnn.add_module('conv1', nn.Conv2d(channel, 32, 3, padding=1))
        self.pan_cnn.add_module('relu1', nn.LeakyReLU(1e-2,True))
        self.pan_cnn.add_module('conv2', nn.Conv2d(32, 32, 3, padding=1))
        self.pan_cnn.add_module('relu2', nn.LeakyReLU(1e-2,True))
        self.pan_cnn_down =  nn.Conv2d(32, 64, 2, stride=2)
        
        self.fusion = nn.Sequential()
        self.fusion.add_module('conv4', nn.Conv2d(128,128,3,padding=1))
        self.fusion.add_module('relu4', nn.LeakyReLU(1e-2,True))
        self.fusion.add_module('conv5', nn.Conv2d(128,128,3,padding=1))
        self.fusion.add_module('relu5', nn.LeakyReLU(1e-2,True))
        self.fusion_down =  nn.Conv2d(128,256,2,stride=2)

        self.res1 = nn.Sequential()
        self.res1.add_module('conv7', nn.Conv2d(256,256,1))
        self.res1.add_module('relu7', nn.LeakyReLU(1e-2,True))
        self.res1.add_module('conv8', nn.Conv2d(256,256,3,padding=1))
        self.res1.add_module('relu8', nn.LeakyReLU(1e-2,True))
        self.res1.add_module('conv9', nn.ConvTranspose2d(256,128,2,stride=2))
        self.res1.add_module('relu9', nn.LeakyReLU(1e-2,True))

        self.res2 = nn.Sequential()
        self.res2.add_module('conv10', nn.Conv2d(256,128,3,padding=1))
        self.res2.add_module('relu10', nn.LeakyReLU(1e-2,True))
        self.res2.add_module('conv11', nn.ConvTranspose2d(128,64,2,stride=2))
        self.res2.add_module('relu11', nn.LeakyReLU(1e-2,True))

        self.res3 = nn.Sequential()
        self.res3.add_module('conv12', nn.Conv2d(128,64,3,padding=1))
        self.res3.add_module('relu12', nn.LeakyReLU(1e-2,True))
        self.res3.add_module('conv13', nn.Conv2d(64,Channel,3,padding=1))

    def forward(self, ms, pan):
        ms = self.ms_cnn(ms)
        ms_2 = self.leaky_relu(self.ms_cnn_down(ms))
        pan = self.pan_cnn(pan)
        pan_2 = self.leaky_relu(self.pan_cnn_down(pan))
        
        feat = torch.cat([ms_2,pan_2], dim=1)
        feat = self.fusion(feat)
        feat_4 = self.leaky_relu(self.fusion_down(feat))
        
        feat_2 = self.res1(feat_4)
        feat_2 = torch.cat([feat,feat_2], dim=1)
        
        feat = self.res2(feat_2)
        feat = torch.cat([feat,ms,pan], dim=1)
        
        return self.res3(feat)

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
learning_rate = 1e-4

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
net = TFN(*n_Channel).to('cuda')
optimizer    = optim.Adam(net.parameters(), lr=learning_rate)
lossfun = nn.MSELoss()

# logs
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('TFN_logs',timestamp+'_bs%d_epoch%d_lr%.5f_scale%d_%s'%(batch_size,num_epoch,learning_rate,scale,data_name))
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
        gt_hat       = net(lr,guide)
        loss         = lossfun(gt,gt_hat)
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
            imgf = torch.clamp(net(lr,guide), 0., 1.)
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
        imgf = torch.clamp(net(lr, guide), 0., 1.)
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
