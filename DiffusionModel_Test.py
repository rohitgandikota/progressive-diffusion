#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:03:26 2022

@author: Rohit Gandikota
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion



models = glob.glob('/appdisk/projects/Warhol/run/diffusionModel/human/*.pt')
models.reverse()
for model_file in models:
    mod = Unet(dim = 128, dim_mults = (1, 2, 4, 8)).cuda()
    diff = GaussianDiffusion(mod,image_size = 128,timesteps = 1000, loss_type = 'l1').cuda()
    model_check = torch.load(model_file)
    model_dict = model_check['model']
    diff.load_state_dict(model_dict)
    sampled_images = diff.sample(batch_size = 50)
    a = np.array(sampled_images.cpu())
    a = np.einsum('ijkl->iklj',a)
    for i in range(len(a)):
        fig=plt.figure(figsize=(5,5))
        plt.imshow(a[i])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'/appdisk/projects/Warhol/run/diffusionModel/cat/testImages/Sample-{i}_{os.path.basename(model_file)}.png')