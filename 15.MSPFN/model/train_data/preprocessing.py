import os
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import requests
import tarfile

def en_sure(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def preprocess():
    print('... loading data')
    # os.mkdir('train_data/raw')
    # os.mkdir('train_data/raw/train_rain')
    # os.mkdir('train_data/raw/test_rain')
    # os.mkdir('train_data/npy')
    en_sure('train_data/raw')
    en_sure('train_data/raw/train_rain')
    en_sure('train_data/raw/train_gt')
    en_sure('train_data/npy')

    rain_files = sorted(glob.glob('C:/DATASETS/Heavy_rain_image_cvpr2019/val_301_350/in/'))####rainysamples
    clean_files = sorted(glob.glob('C:/DATASETS/Heavy_rain_image_cvpr2019/val_301_350/gt/'))
    rain_paths = np.array(
        [e for x in [glob.glob(os.path.join(file, '*'))
        for file in rain_files] for e in x])
    clean_paths = np.array(
        [e for x in [glob.glob(os.path.join(file, '*'))
        for file in clean_files] for e in x])
    #np.random.shuffle(paths)

    # r = int(len(paths) * 0.999)
    # train_paths = paths[:r]
    # test_paths = paths[r:]

    x_rain = []
    pbar = tqdm(total=(len(rain_paths)))
    for i, d in enumerate(rain_paths):
        pbar.update(1)
        img = cv2.imread(d)
        img = cv2.resize(img, (96, 96))#128
        if img is None:
            continue
        x_rain.append(img)
        name = "{}.png".format("{0:05d}".format(i))
        imgpath = os.path.join('train_data/raw/train_rain', name)
        cv2.imwrite(imgpath, img)
    pbar.close()

    x_clean = []
    pbar = tqdm(total=(len(clean_paths)))
    for i, d in enumerate(clean_paths):
        pbar.update(1)
        img = cv2.imread(d)
        img = cv2.resize(img, (96, 96))#128
        if img is None:
            continue
        for j in range(15):
            x_clean.append(img)
            name = "{}.png".format("{0:05d}".format(i*15+j))
            imgpath = os.path.join('train_data/raw/train_gt', name)
            cv2.imwrite(imgpath, img)

    pbar.close()

    x_rain = np.array(x_rain, dtype=np.uint8)
    x_clean = np.array(x_clean, dtype=np.uint8)
    np.save('train_data/npy/train_rain.npy', x_rain)
    np.save('train_data/npy/train_gt.npy', x_clean)
    #np.save('train_data/npy/train_rain.npy', x_train)
    #np.save('train_data/npy/test_rain.npy', x_test)

def main():
    preprocess()


if __name__ == '__main__':
    main()

