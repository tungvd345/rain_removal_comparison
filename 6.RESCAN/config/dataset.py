import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset

import settings 


class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        # self.root_dir = os.path.join(settings.data_dir, name)
        if name == 'val':
            self.root_dir = os.path.join(settings.data_dir, 'val_301_350')
        elif name == 'train':
            self.root_dir = os.path.join(settings.data_dir, 'train_301_350')

        # self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
        # self.file_num = len(self.mat_files)
        self.root_dir_in = os.path.join(self.root_dir, 'in')
        self.root_dir_tar = os.path.join(self.root_dir, 'gt')
        self.mat_files_in = os.listdir(self.root_dir_in)
        self.mat_files_tar = os.listdir(self.root_dir_tar)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files_in)


    def __len__(self):
        # return self.file_num * 100
        return self.file_num

    def __getitem__(self, idx):
        # file_name = self.mat_files[idx % self.file_num]
        # img_file = os.path.join(self.root_dir, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255
        file_name_in = self.mat_files_in[idx]
        file_name_tar = self.mat_files_tar[idx//15]
        img_in = cv2.imread(os.path.join(self.root_dir_in, file_name_in)).astype(np.float32)/255
        img_tar = cv2.imread(os.path.join(self.root_dir_tar, file_name_tar)).astype(np.float32)/255

        if settings.aug_data:
            # O, B = self.crop(img_pair, aug=True)
            O, B = self.crop(img_in, img_tar, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
        else:
            # O, B = self.crop(img_pair, aug=False)
            O, B = self.crop(img_in, img_tar, aug=True)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample

    # def crop(self, img_pair, aug):
    def crop(self, img_in, img_tar, aug):
        patch_size = self.patch_size
        # h, ww, c = img_pair.shape
        # w = int(ww / 2)
        h, w, c = img_in.shape

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        # O = img_pair[r: r+p_h, c+w: c+p_w+w]
        # B = img_pair[r: r+p_h, c: c+p_w]
        O = img_in[r: r+p_h, c: c+p_w]
        B = img_tar[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        # self.root_dir = os.path.join(settings.data_dir, name)
        # self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
        # self.file_num = len(self.mat_files)

        # self.root_dir = os.path.join(settings.data_dir, 'test_with_train_param_v5')
        self.root_dir = 'C:/DATASETS/real_rain'
        self.root_dir_in = os.path.join(self.root_dir, 'in')
        self.root_dir_tar = os.path.join(self.root_dir, 'gt')
        self.mat_files_in = os.listdir(self.root_dir_in)
        self.mat_files_tar = os.listdir(self.root_dir_tar)
        self.file_num = len(self.mat_files_in)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        # file_name = self.mat_files[idx % self.file_num]
        # img_file = os.path.join(self.root_dir, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255
        # h, ww, c = img_pair.shape
        # w = int(ww / 2)

        file_name_in = self.mat_files_in[idx]
        # file_name_tar = self.mat_files_tar[idx//15] # syn data
        file_name_tar = self.mat_files_tar[idx]       # real data  
        img_in = cv2.imread(os.path.join(self.root_dir_in, file_name_in)).astype(np.float32)/255
        img_tar = cv2.imread(os.path.join(self.root_dir_tar, file_name_tar)).astype(np.float32)/255

        # O = np.transpose(img_pair[:, w:], (2, 0, 1))
        # B = np.transpose(img_pair[:, :w], (2, 0, 1))
        O = np.transpose(img_in, (2, 0, 1))
        B = np.transpose(img_tar, (2, 0, 1))

        sample = {'O': O, 'B': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        # self.root_dir = os.path.join(settings.data_dir, name)
        # self.img_files = sorted(os.listdir(self.root_dir))
        # self.file_num = len(self.img_files)

        # self.root_dir = os.path.join(settings.data_dir, 'test_with_train_param_v5')
        self.root_dir = 'D:/DATASETS/real_rain/new'
        self.root_dir_in = os.path.join(self.root_dir, 'in')
        self.root_dir_tar = os.path.join(self.root_dir, 'gt')
        self.mat_files_in = sorted(os.listdir(self.root_dir_in))
        self.mat_files_tar = sorted(os.listdir(self.root_dir_tar))
        self.file_num = len(self.mat_files_in)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        # file_name = self.img_files[idx % self.file_num]
        # img_file = os.path.join(self.root_dir, file_name)
        # img_pair = cv2.imread(img_file).astype(np.float32) / 255

        # h, ww, c = img_pair.shape
        # w = int(ww / 2)

        # O = np.transpose(img_pair[:, w:], (2, 0, 1))
        file_name_in = self.mat_files_in[idx % self.file_num]
        # file_name_tar = self.mat_files_tar[(idx % self.file_num)//15] # syn data
        file_name_tar = self.mat_files_tar[(idx % self.file_num)] # real data
        img_in = cv2.imread(os.path.join(self.root_dir_in, file_name_in)).astype(np.float32)/255
        img_tar = cv2.imread(os.path.join(self.root_dir_tar, file_name_tar)).astype(np.float32)/255

        O = np.transpose(img_in, (2, 0, 1))
        sample = {'O': O, 'idx': idx}

        return sample

    def get_name(self, idx):
        return self.mat_files_in[idx % self.file_num].split('.')[0]


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    dt = TestDataset('test')
    print('TestDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
