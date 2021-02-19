import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from randomcrop import RandomRotation,RandomResizedCrop,RandomHorizontallyFlip,RandomVerticallyFlip
import PIL.Image as Image

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        # self.mat_files = open(self.dataset,'r').readlines()

        self.in_path = self.dataset + '/in'
        self.mat_in_files = sorted(os.listdir(self.in_path))
        self.gt_path = self.dataset + '/gt'
        self.mat_gt_files = sorted(os.listdir(self.gt_path))
        self.file_num = len(self.mat_in_files)
        self.rc = RandomResizedCrop(240)

    def __len__(self):
        return self.file_num * 100
        # return self.file_num

    def __getitem__(self, idx):
        # file_name = self.mat_files[idx % self.file_num]
        # gt_file = file_name.split(' ')[1][:-1]
        # img_file = file_name.split(' ')[0]
        gt_file = self.mat_gt_files[(idx%self.file_num) // 15]
        img_file = self.mat_in_files[idx%self.file_num]

        O = cv2.imread(os.path.join(self.in_path, img_file))
        B = cv2.imread(os.path.join(self.gt_path, gt_file))

        O = Image.fromarray(O)
        B = Image.fromarray(B)

        O,B = self.rc(O,B)
        O,B = np.array(O),np.array(B)

        M = np.clip((O-B).sum(axis=2),0,1).astype(np.float32)
        O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
        B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1)) 

        sample = {'O': O, 'B': B,'M':M}

        return sample



class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        # self.mat_files = open(self.root_dir,'r').readlines()

        self.in_path = self.root_dir + '/in'
        self.mat_in_files = sorted(os.listdir(self.in_path))
        self.gt_path = self.root_dir + '/gt'
        self.mat_gt_files = sorted(os.listdir(self.gt_path))
        self.file_num = len(self.mat_in_files)
        # self.file_num = len(self.mat_files)
        
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        # file_name = self.mat_files[idx % self.file_num]
        # gt_file = "." + file_name.split(' ')[1][:-1]
        # img_file = "." + file_name.split(' ')[0]

        # gt_file = self.mat_gt_files[(idx%self.file_num) // 15] # syn data
        gt_file = self.mat_gt_files[(idx%self.file_num)]        # real data
        img_file = self.mat_in_files[idx%self.file_num]
        
        O = cv2.imread(os.path.join(self.in_path, img_file))
        B = cv2.imread(os.path.join(self.gt_path, gt_file))

        M = np.clip((O-B).sum(axis=2),0,1).astype(np.float32)
        O = np.transpose(O, (2, 0, 1)).astype(np.float32) / 255.0 
        B = np.transpose(B, (2, 0, 1)).astype(np.float32) / 255.0 

        sample = {'O': O,'B':B,'M':M, 'name':img_file}

        return sample
