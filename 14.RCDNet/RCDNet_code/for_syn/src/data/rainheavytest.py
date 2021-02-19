import os
from data import srdata

class RainHeavyTest(srdata.SRData):
    def __init__(self, args, name='RainHeavyTest', train=True, benchmark=False):
        super(RainHeavyTest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(RainHeavyTest, self)._scan()
        # names_hr = names_hr[self.begin - 1:self.end]
        # names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        # names_hr = names_hr[0:3]
        # names_hr = names_hr[0:45]
        # names_lr = [n[0:45] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainHeavyTest, self)._set_filesystem(dir_data)
        # self.apath = '../data/test/small/'
        # self.apath = 'C:/DATASETS/Heavy_rain_image_cvpr2019/val_301_350'
        self.apath = 'C:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param_v5'
        # self.apath = 'D:/DATASETS/JORDER_DATASET/test/rain_data_test_Heavy/rain_heavy_test'
        print(self.apath)
        # self.dir_hr = os.path.join(self.apath, 'norain')
        # self.dir_lr = os.path.join(self.apath, 'rain')
        self.dir_hr = os.path.join(self.apath, 'gt')
        self.dir_lr = os.path.join(self.apath, 'in')

