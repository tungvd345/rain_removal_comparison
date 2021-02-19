# UMRL--using-Cycle-Spinning
Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining

[Rajeev Yasarla](https://sites.google.com/view/rajeevyasarla/home), [Vishal M. Patel](https://engineering.jhu.edu/ece/faculty/vishal-m-patel/)

[Paper Link](https://arxiv.org/abs/1906.11129) (CVPR'19)


    @InProceedings{Yasarla_2019_CVPR,
        author = {Yasarla, Rajeev and Patel, Vishal M.},
        title = {Uncertainty Guided Multi-Scale Residual Learning-Using a Cycle Spinning CNN for Single Image De-Raining},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }

We present a novel Uncertainty guided Multi-scale Residual Learning (UMRL) network to address the single image de-raining. The proposed network attempts to  address this issue by learning the rain content at different scales and using them to estimate the final de-rained output.  In addition, we introduce a technique which guides the network to learn the network weights based on the confidence measure about the estimate.  Furthermore, we introduce a new training and testing procedure based on the notion of cycle spinning to improve the final de-raining performance.

## Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)

## To test UMRL:
python umrl_test.py --dataroot ./facades/validation --valDataroot ./facades/validation --netG ./pre_trained/Net_DIDMDN.pth

## To train UMRL:
python umrl_train.py  --dataroot <dataset_path>  --valDataroot ./facades/validation --exp ./check --netG ./pre_trained/Net_DIDMDN.pth

## To test UMRL using Cycle Spining:
python umrl_cycspn_test.py --dataroot ./facades/validation --valDataroot ./facades/validation --netG ./pre_trained/Net_DIDMDN.pth

## To train UMRL using Cycle Spining:
python umrl_cycspn_train.py  --dataroot <dataset_path>  --valDataroot ./facades/validation --exp ./check --netG ./pre_trained/Net_DIDMDN.pth

## Acknowledgments
Thanks for the discussions with, and help from [He Zhang](https://sites.google.com/site/hezhangsprinter/)
