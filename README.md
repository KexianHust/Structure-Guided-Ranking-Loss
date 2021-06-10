# Structure-Guided Ranking Loss for Single Image Depth Prediction
This repository contains a pytorch implementation of our CVPR2020 paper "Structure-Guided Ranking Loss for Single Image Depth Prediction". 
[Project Page](https://KexianHust.github.io/Structure-Guided-Ranking-Loss/)
![Teaser Image](https://KexianHust.github.io/Structure-Guided-Ranking-Loss/teaser.png)

## Changelog
* [Jun. 2020] Initial release

## To do
- [ ] Mix data training

## Prerequisites
* Pytorch >= 0.4.1
* CUDA >= 0.8
* Python >= 2.7
* glob, matplotlib
* Need to compile the syncbn module in models/syncbn. Note that the directory of the syncbn module should be modified in some .py files (i.e., DepthNet.py, resnet.py and networks.py)
* Download the [model.pth.tar](https://drive.google.com/file/d/1p8c8-nUTNry5usQmGdTC2TrwWrp3dQ0y/view?usp=sharing)

## Inference
```bash
# Before running, you should set the CUDA_VISIBLE_DEVICES in demo.sh
bash demo.sh

```

If you find our work useful in your research, please consider citing the paper.

```
@InProceedings{Xian_2020_CVPR,
author = {Xian, Ke and Zhang, Jianming and Wang, Oliver and Mai, Long and Lin, Zhe and Cao, Zhiguo},
title = {Structure-Guided Ranking Loss for Single Image Depth Prediction},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Dataset
Our [HRWSI](https://drive.google.com/file/d/1OVOx6x-B0Cs-m2z_-7ZxSgRFHz_VBvDd/view?usp=sharing) dataset is for research only! Some researchers may interested in the stereo data, so we provide the right views [here](https://drive.google.com/file/d/1HzEB7yQI05Q21dP9rRjnyMoEmvCckAQp/view?usp=sharing). Please let me know if you have any questions.

## Lisence
Research only
