# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:34:49 2020

@author: win10
"""

import torch
import torch.nn as nn
import block as B
import torch.nn.functional as F

def tensor_resize(tensor, upscale):
    return F.upsample(tensor, scale_factor=upscale, mode='bilinear', align_corners=True)

class PNN(nn.Module):
    """Pansharpening by Convolutional Neural Networks"""

    def __init__(self,
                 in_nc=5,
                 out_nc=4,
                 nf=64,
                 nb=None,
                 upscale=4,
                 norm_type=None,
                 act_type='relu',
                 mode='CNA',
                 upsample_mode='upconv'):
        super().__init__()
        self.upscale = upscale
        conv1 = B.conv_block(in_nc,
                             nf,
                             kernel_size=9,
                             norm_type=None,
                             act_type=act_type)
        conv2 = B.conv_block(nf,
                             nf // 2,
                             kernel_size=5,
                             norm_type=None,
                             act_type=act_type)
        conv3 = B.conv_block(nf // 2,
                             out_nc,
                             kernel_size=5,
                             norm_type=None,
                             act_type=None)

        self.model = B.sequential(conv1, conv2, conv3)

    def forward(self, x, p=None):
        # resize to the same size as PAN
        # (bs, c, h, w)
        if p is None:
            raise TypeError('Pan image must be supplied!')
        x = tensor_resize(x, upscale=self.upscale)
        x = torch.cat((x, p), dim=1)
        x = self.model(x)
        return x

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
net = PNN(in_nc=31+3, out_nc=31,upscale=scale).cuda()
optimizer    = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
lossfun = nn.MSELoss()

# logs
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join('PNN_logs',timestamp+'_bs%d_epoch%d_lr%.5f_scale%d_%s'%(batch_size,num_epoch,learning_rate,scale,data_name))
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
        imgf = torch.clamp(net(lr,guide), 0., 1.)
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
