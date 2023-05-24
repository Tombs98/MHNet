import os
import numpy as np
from glob import glob
import cv2
from natsort import natsorted

from skimage.metrics import structural_similarity,peak_signal_noise_ratio

def read_img(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def ssim(tf_img1, tf_img2):
    return structural_similarity(tf_img1,tf_img2)

def main():
    datasets = {'GoPr', 'HIDE'};
    file_path = os.path.join('results_mhnet', 'HIDE')
    gt_path = os.path.join('Datasets/HIDE/test', 'target')
    print(file_path)
    print(gt_path)

    path_fake = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    path_real = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    print(len(path_fake))
    list_psnr = []
    list_ssim = []
    list_mse = []

    for i in range(len(path_real)):
        t1 = read_img(path_real[i])
        t2 = read_img(path_fake[i])
        result1 = np.zeros(t1.shape,dtype=np.float32)
        result2 = np.zeros(t2.shape,dtype=np.float32)
        cv2.normalize(t1,result1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        cv2.normalize(t2,result2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        ssim_num = ssim(result1, result2)
        list_ssim.append(ssim_num)
  

    print("AverSSIM:", np.mean(list_ssim))  # ,list_ssim)
   

if __name__ == '__main__':
    main()
