#! bash

# Rain100H
#python train_PReNet.py --preprocess True --save_path logs/HeavyRain/PReNet --data_path D:/DATASETS/Heavy_rain_image_cvpr2019/LRBI/train #datasets/train/RainTrainH
#python train_PReNet.py --save_path logs/HeavyRain/PReNet
python train_PReNet.py --save_path logs/HeavyRain/PReNet1e-6 --lr 1e-6


# Rain100L
# python train_PReNet.py --preprocess True --save_path logs/Rain100L/PReNet --data_path datasets/train/RainTrainL

# Rain12600
# python train_PReNet.py --preprocess True --save_path logs/Rain1400/PReNet --data_path datasets/train/Rain12600
