import os
from data import srdata

class RainHeavy(srdata.SRData):
    def __init__(self, args, name='RainHeavy', train=True, benchmark=False):
        super(RainHeavy, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(RainHeavy, self)._scan()
        # names_hr = names_hr[self.begin - 1:self.end]
        # names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavy, self)._set_filesystem(dir_data)
        # self.apath = '../data/train/small/'
        # self.apath = 'C:/DATASETS/Heavy_rain_image_cvpr2019/train_301_350'
        self.apath = 'D:/DATASETS/JORDER_DATASET/train/rain_data_train_Heavy'
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')
        # self.dir_hr = os.path.join(self.apath, 'gt')
        # self.dir_lr = os.path.join(self.apath, 'in')
