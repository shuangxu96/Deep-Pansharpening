# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:22:36 2020

@author: BSawa
"""
import os
import h5py
import torch

from glob import glob
import numpy as np
import torch.utils.data as Data
    
'''
Dataset & Image Pre-processing
'''

def im2double(img):
    if img.dtype=='uint8':
        img = img.astype(np.float32)/255.
    elif img.dtype=='uint16':
        img = img.astype(np.float32)/65535.
    else:
        img = img.astype(np.float32)
    return img

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def imresize(img, size=None, scale_factor=None):
    # img (np.array) - [C,H,W]
    imgT = torch.from_numpy(img).unsqueeze(0) #[1,C,H,W]
    if size is None and scale_factor is not None:
        imgT = torch.nn.functional.interpolate(imgT, scale_factor=scale_factor)
    elif size is not None and scale_factor is None:
        imgT = torch.nn.functional.interpolate(imgT, size=size)
    else:
        print('Neither size nor scale_factor is given.')
    imgT = imgT.squeeze(0).numpy()
    return imgT
    
def prepare_data(data_path, 
                 patch_size, 
                 aug_times=4,
                 stride=25, 
                 synthetic=True, 
                 scale=2,
                 file_name='train.h5'
                 ):
    # patch_size : the window size of low-resolution images
    # scale : the spatial ratio between low-resolution and guide images
    # train
    print('process training data')
    files = glob(os.path.join(data_path, 'train', '*.mat'))
    h5f = h5py.File(file_name, 'w')
    h5gt = h5f.create_group('GT')
    h5guide = h5f.create_group('Guide')
    h5lr = h5f.create_group('LR')
    train_num = 0
    for i in range(len(files)):
        img = h5py.File(files[i])
        lr = im2double(img['LR'][:])
        guide = im2double(img['Guide'][:])
        
        if synthetic:
            # if synthetic is True: the spatial resolutions of lr and guide are the same
            lr_patches = Im2Patch(lr, win=scale*patch_size, stride=stride) #[C,H,W,N]
            guide_patches = Im2Patch(guide, win=scale*patch_size, stride=stride)
        else:
            scale = int(guide.shape[-1]/lr.shape[-1])
            guide = imresize(guide, size=lr.shape[1:])
            lr_patches = Im2Patch(lr, win=scale*patch_size, stride=stride) #[C,H,W,N]
            guide_patches = Im2Patch(guide, win=scale*patch_size, stride=stride)
            
        print("file: %s # samples: %d" % (files[i], lr_patches.shape[3]*aug_times))
        for n in range(lr_patches.shape[3]):
            gt_data = lr_patches[:,:,:,n].copy()
            guide_data = guide_patches[:,:,:,n].copy()
            lr_data = imresize(gt_data, scale_factor=1/scale)
            
            h5gt.create_dataset(str(train_num), 
                                data=gt_data, dtype=gt_data.dtype,shape=gt_data.shape)
            h5guide.create_dataset(str(train_num), 
                                   data=guide_data, dtype=guide_data.dtype,shape=guide_data.shape)
            h5lr.create_dataset(str(train_num), 
                                data=lr_data, dtype=lr_data.dtype,shape=lr_data.shape)
            train_num += 1
            for m in range(aug_times-1):
                gt_data_aug = np.rot90(gt_data, m+1, axes=(1,2))
                guide_data_aug = np.rot90(guide_data, m+1, axes=(1,2))
                lr_data_aug = np.rot90(lr_data, m+1, axes=(1,2))
                
                h5gt.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=gt_data_aug, dtype=gt_data_aug.dtype,shape=gt_data_aug.shape)
                h5guide.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                       data=guide_data_aug, dtype=guide_data_aug.dtype,shape=guide_data_aug.shape)
                h5lr.create_dataset(str(train_num)+"_aug_%d" % (m+1), 
                                    data=lr_data_aug, dtype=lr_data_aug.dtype,shape=lr_data_aug.shape)
                train_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['Guide'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        guide = np.array(h5f['Guide'][key])
        gt    = np.array(h5f['GT'][key])
        lr    = np.array(h5f['LR'][key])
        h5f.close()
        return torch.Tensor(lr),torch.Tensor(guide),torch.Tensor(gt)

class MMSRDataset(Data.Dataset):
    def __init__(self, root, scale):
        self.scale = scale
        self.root = root
        self.files = glob(root+'/*.mat')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        temp  = h5py.File(self.files[index])
        guide = im2double(temp['Guide'][:])
        gt    = im2double(temp['LR'][:])
        lr    = imresize(gt, scale_factor=1/self.scale)
        del temp
        return lr, guide, gt


'''
Others
'''
def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
