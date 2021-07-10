import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
import cv2
import os, glob, numpy as np
from os.path import join as osj
from skimage.restoration import denoise_wavelet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.utils import save_image

def sorting_truncated(fn):
    len1=len(fn)
       
    fns = []
    basename = '/'.join(fn[0].split('/')[:-1])

    for i in range(len(fn)):
        path=osj(basename, str(i)+'.jpg')
        if os.path.isfile(path):
            fns.append(path)
#         else:
#             print('Something wrong!!', path)
    return fns
def check_fn(filename):
    assert os.path.isfile(filename)


class dataset_DFD(torch.utils.data.Dataset):
    
    def __init__(self, local_rank, args, mode='train', file_list=None, root=None):
        super(dataset_DFD, self).__init__()

        self.mode        = mode
        self.root        = args.dataset_dir if mode!='test' and mode!='finetuning' else args.dataset_dir_test
        self.FRR         = args.FRR
        self.FREQ        = args.FREQ
        self.seq_len     = []
        self.fns         = []
        self.image_size  = args.image_size
        self.crop_size   = args.crop_size
        self.offsets     = (args.image_size-args.crop_size)//2
        self.random_crop = True
        self.marginal    = args.marginal if mode!='test' else 0
        self.label       = []
        self.local_rank  = local_rank
        self.test_aug    = args.test_aug
        self.max_det     = args.max_det
        self.FSet        = [int(v) for v in args.MultiFREQ.split(',')] if args.MultiFREQ is not None else None
        filename         = args.train_file if mode!='test' else args.test_file
        self.maxFREQ     = args.FRR if (self.FSet is None or mode=='test') else max(self.FSet)
        self.centerCrop  = args.centerCrop/100.0
        self.twoStream   = args.twoStream
        self.allRandRoate= args.allRandRoate
        self.singleAug   = args.singleAug
        self.evalPerformance= args.evalPerformance
        
        if root is not None:
            self.root = root
        
        ## Fake one
        Y=None
        
        Y=[]
        if file_list is not None:
            data = file_list[:]
        else:
            print('Preparing the file from', filename)
            with open(filename, 'r') as fp:
                data = fp.readlines()
                
        file_list = []


        for line in data:
            fn, lab=line.split(' ')
            fn_list = glob.glob(osj(self.root, fn, '*.jpg'))
            fn_list = sorting_truncated(fn_list)
         
            if len(fn_list)<self.FRR:
                print('Pass the filename', fn, 'due to inefficient number of training samples: target:', self.FRR, 'source:', len(fn_list))
#                 

            self.label.append(int(lab.strip('\n')))
            file_list.append(fn_list)
            self.seq_len.append(len(fn_list))
            self.fns.append(self.root+fn)


        self.testaug = A.Compose([
                                A.Resize(self.image_size, self.image_size),
                                A.CenterCrop(self.crop_size, self.crop_size)
                                ], p=1)
        
        
       
        self.image_list = file_list

        self.n_images = len(self.image_list)
        if local_rank==0:
            print('The # of videos is:', self.n_images, 'on', self.mode, 'mode!')
            
    def randRotate(self, x):
        if rn.random()>=0.5:
            x = x[:, ::-1, :, :].copy()
        if rn.random()>=0.5:
            x = x[:, :, ::-1, :].copy()   
        return x
    
    def cc(self, x):
        len1 = len(x)
        FREQ = self.FREQ
        perc = self.centerCrop
        targetLen = int(len1 * perc)
        while(self.FRR > targetLen):
            perc += 0.1
            targetLen = int(len1 * perc)
            if perc>1:
                raise
                break


        if len1 > targetLen:
            rem = len1 - targetLen
            rem = int(np.floor(rem/2))
            x = [x[ind] for ind in range(rem,rem+targetLen)]
            
        return x
    
    def transform(self, img_list, npfile, len1, inner_transform=None, fn=None):
        data, label = {}, {}
        
        imgs = []
       
        
        x = []
        for ind, ims in enumerate(img_list):
            assert os.path.isfile(ims)
            im = cv2.imread(ims)
            if im is None:
                continue
            if ind>0:
                im = cv2.resize(im, dsize=(col, row))
            else:
                row, col, c=im.shape
            x.append(im)
        
        len1 = len(x)
        
        if len1<self.FRR:
            x2 = np.zeros((self.FRR, x.shape[1], x.shape[2], x.shape[3]), np.uint8)
            x2[:len(x)] = x
            x2[len(x):] = x[-1]
            x=x2
            
            assert len(x)==self.FRR

            
        FREQ = self.FREQ 
        if self.test_aug==0:
            
            nStep = int(np.floor((len1-1) / self.FRR))
            max_ind = nStep * self.FRR
            max_ind = len1 if max_ind>=len1-1 else max_ind
            ind = list(range(0, max_ind+1, nStep))
            if len(ind)<self.FRR:
                ind = range(self.FRR)
            x = [x[index] for index in ind ]
            
        elif self.test_aug==1:  # K-crops data augmentation in testing phase
            max_det=self.max_det
            assert len1>=self.FRR
            while len1< self.FRR*FREQ:
                FREQ-=1 
                if FREQ==1:
                    break
                
            len11=len1
            targetLen = self.FRR*FREQ + max_det*FREQ
            if len1 > targetLen:
                rem = len1 - targetLen
                rem = int(np.floor(rem/2))
                x = [x[ind] for ind in range(rem,rem+targetLen)]
            else:
                x = [x[ind] for ind in range(len1)]

            len1 = len(x)
            
        elif self.test_aug==2:  # center-crops data augmentation in testing phase
            if self.centerCrop>0 and len1 > self.FRR / self.centerCrop:
                x = self.cc(x)

            len1 = len(x)
            
                

        ## Center crop
        x2 = []
        for im in x:
            x2.append(inner_transform(image=im)['image'])
        x = (np.array(x2) - 127.5)/128.0 


        x = torch.Tensor(x).permute(0, 3, 1, 2)
        data['img']={}
        data['fn'] =fn
        for ind, item in enumerate(x):
            data['img'][ind] = item
            
        return data
        
    def __getitem__(self, index):
        img_list = self.image_list[index]
        len1 = self.seq_len[index]
        fn = self.fns[index]

        im = self.transform(img_list, fn, len1, inner_transform=self.testaug, fn=fn)
        lab = self.label[index]

        if self.evalPerformance:
            return im, lab
        else:
            return im

    def __len__(self):
        return self.n_images
    
