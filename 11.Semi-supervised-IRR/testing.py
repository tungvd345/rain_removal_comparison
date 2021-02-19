import os
import training as Network
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import skimage.measure
import argparse

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
############################################################################
#os.environ['CUDA_VISIBLE_DEVICES'] = str(monitoring_gpu.GPU_INDEX)
############################################################################

# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('rain_path', "D:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param_v5/in/im_0601_s80_a04.png","path to rain image")

parser = argparse.ArgumentParser(description='semi rain removal - 2019')
parser.add_argument('--rain_path', type=str, help='path to rain image', default='D:/DATASETS/Heavy_rain_image_cvpr2019/test_with_train_param_v5/in/')
parser.add_argument('--rain_name', type=str, help='name of rain image', default='im_0601_s80_a06.png')

args = parser.parse_args()


def guided_filter(data, height,width):
    r = 15
    eps = 1.0
    batch_size = 1
    channel = 3
    batch_q = np.zeros((batch_size, height, width, channel))
    for i in range(batch_size):
        for j in range(channel):
            I = data[i, :, :,j] 
            p = data[i, :, :,j] 
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q
    return batch_q

if __name__ == '__main__':
    # file = "./test-image/input.png"
    # file = args.rain_path + args.rain_name
    list_file = os.listdir(args.rain_path)
    # print(list_file)
    # for idx in range(len(list_file)):
    # file_name = list_file[idx]
    file_path = args.rain_path + args.rain_name
    ori = img.imread(file_path)
    ori = np.double(ori)

    input = np.expand_dims(ori[:,:,:], axis = 0)
    detail_layer = input - guided_filter(input, input.shape[1], input.shape[2])


    num_channels = 3
    image = tf.compat.v1.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))
    detail = tf.compat.v1.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))

    output = Network.inference(image,detail,factor=1.)
    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, ckpt)
        print ("Loading model")

        final_output  = sess.run(output, feed_dict={image:input, detail: detail_layer})

        final_output[np.where(final_output < 0. )] = 0.
        final_output[np.where(final_output > 1. )] = 1.
        derained = final_output[0,:,:,:]

        # plt.subplot(1,2,1)
        # plt.imshow(ori)
        # plt.title('input')

        # plt.subplot(1,2,2)
        # plt.imshow(derained)
        # plt.title('output')
        result_path = './test_results/' + args.rain_name
        # plt.imsave('./test-image/output.png',derained)
        plt.imsave(result_path, derained)

        # psnr = skimage.measure.compare_psnr(np.uint8(255*img.imread('./test-image/norain.png')),np.uint8(255*derained))
        # print(psnr)

        # plt.title('output')
        # plt.show()
