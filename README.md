

# Mixed Hierarchy Network for Image Restoration

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](http://arxiv.org/abs/2302.09554)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixed-hierarchy-network-for-image-restoration/image-deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/image-deblurring-on-hide-trained-on-gopro?p=mixed-hierarchy-network-for-image-restoration)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixed-hierarchy-network-for-image-restoration/single-image-deraining-on-test1200)](https://paperswithcode.com/sota/single-image-deraining-on-test1200?p=mixed-hierarchy-network-for-image-restoration)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixed-hierarchy-network-for-image-restoration/single-image-deraining-on-test100)](https://paperswithcode.com/sota/single-image-deraining-on-test100?p=mixed-hierarchy-network-for-image-restoration)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixed-hierarchy-network-for-image-restoration/single-image-deraining-on-rain100l)](https://paperswithcode.com/sota/single-image-deraining-on-rain100l?p=mixed-hierarchy-network-for-image-restoration)


<hr />


> **Abstract:** Image restoration is a long-standing low-level vision problem, e.g., deblurring and deraining. In the process of image restoration, it is necessary to consider not only the spatial details and contextual information of restoration to ensure the quality, but also the system complexity. Although many methods have been able to guarantee the quality of image restoration, the system complexity of the state-of-the-art (SOTA) methods is increasing as well. Motivated by this, we present a mixed hierarchy network that can balance these competing goals. Our main proposal is a mixed hierarchy architecture, that progressively recovers contextual information and spatial details from degraded images while we design intra-blocks to reduce system complexity. Specifically, our model first learns the contextual information using encoder-decoder architectures, and then combines them with high-resolution branches that preserve spatial detail. In order to reduce the system complexity of this architecture for convenient analysis and comparison, we replace or remove the nonlinear activation function with multiplication and use a simple network structure.  In addition, we replace spatial convolution with global self-attention for the middle block of encoder-decoder. The resulting tightly interlinked hierarchy architecture, named as MHNet, delivers strong performance gains on several image restoration tasks, including image deraining, and deblurring. 

## Network Architecture
<table>
  <tr>
    <td align="center"> <img  src = "./fig/network.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of MHNet</b></p></td>
  </tr>
    <tr>
    <td align="center"> <img src = "./fig/dau.png" width="500"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Dual attention</b></p></td>
</table>

## Installation
The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch1 python=3.7
conda activate pytorch1
conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```




## Training and Evaluation

Training and Testing codes for deblurring and deraining  are provided in their respective directories.

## Results


<details>
  <summary> <strong>Image Deblurring</strong> (click to expand) </summary>
<table>
  <tr>
    <td> <img src = "./fig/derain.png" width="450"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Deblurring on GoPro and HIDE Datasets.</b></p></td>
  </tr>
</table></details>

<details>
  <summary> <strong>Image Deraining</strong> (click to expand) </summary>
<img src = "./fig/deblur.png" width="900"></details>

 ## Citations
If our code helps your research or work, please consider citing our paper.
The following is a BibTeX reference:

```
@article{gao2023mixed,
  title={Mixed Hierarchy Network for Image Restoration},
  author={Gao, Hu and Dang, Depeng},
  journal={arXiv preprint arXiv:2302.09554},
  year={2023}
}


```




## Contact
Should you have any question, please contact two_bits@163.com

