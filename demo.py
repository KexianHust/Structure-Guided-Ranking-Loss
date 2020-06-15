#!/usr/bin/env python
# coding: utf-8

# demo

"""
Author: Ke Xian
Email: kexian@hust.edu.cn
Create_Date: 2019/05/21
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
torch.backends.cudnn.deterministic = True
torch.manual_seed(123)

import os, argparse, sys
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

sys.path.append('models')
import DepthNet

# =======================
# demo
# =======================
def demo(net, args):
    data_dir = args.data_dir
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for im in os.listdir(data_dir):
        im_dir = os.path.join(data_dir, im)
        print('Processing img: {}'.format(im_dir))

        # Read image
        img = Image.open(im_dir).convert('RGB')
        ori_width, ori_height = img.size
        int_width = args.img_size[0]
        int_height = args.img_size[1]
        img = img.resize((int_width, int_height), Image.ANTIALIAS)
        tensor_img = img_transform(img)

        # forward
        input_img = torch.autograd.Variable(tensor_img.cuda().unsqueeze(0), volatile=True)
        output = net(input_img)

        # Normalization and save results
        depth = output.squeeze().cpu().data.numpy()
        min_d, max_d = depth.min(), depth.max()
        depth_norm = (depth - min_d) / (max_d - min_d) * 255
        depth_norm = depth_norm.astype(np.uint8)
        image_pil = Image.fromarray(depth_norm)

        output_dir = os.path.join(args.result_dir, im)
        image_pil = image_pil.resize((ori_width, ori_height), Image.BILINEAR)
        plt.imsave(output_dir, np.asarray(image_pil), cmap='inferno')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MRDP Testing/Evaluation')
    parser.add_argument('--img_size', default=[448, 448], type=list, help='Image size of network input')
    parser.add_argument('--data_dir', default='examples', type=str, help='Data path')
    parser.add_argument('--result_dir', default='demo_results', type=str, help='Directory for saving results, default: demo_results')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id, default:0')
    args = parser.parse_args()

    args.checkpoint = 'model.pth.tar'

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    gpu_id = args.gpu_id
    torch.cuda.device(gpu_id)

    net = DepthNet.DepthNet()
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    print('Begin to test ...')
    with torch.no_grad():
        demo(net, args)
    print('Finished!')
