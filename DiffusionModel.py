#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:22:27 2022

@author: Rohit Gandikota
"""
import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def load_real_samples(filename):
    # load dataset
    data = np.load(filename)
    # extract numpy array
    X = data['arr_0']
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = (X + 2) / 2.0
    return X



# model = Unet(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8)
# )

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 128,
#     timesteps = 1000,   # number of steps
#     loss_type = 'l1'    # L1 or L2
# )

# data_path = '/appdisk/projects/Warhol/SpyderVariables/cat_faces.npz'
# training_images = load_real_samples(data_path) # images are normalized from 0 to 1
# training_images = np.einsum('ijkl->iljk',training_images) # change from tensorflow to torch
# training_images = torch.from_numpy(training_images)
# loss = diffusion(training_images)
# loss.backward()
# # after a lot of training

# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)


model = Unet(
    dim = 256,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/home/rohit/wrkspc/data/celeba_hq_256/',
    train_batch_size = 1,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,
    results_folder='/appdisk/projects/Warhol/run/diffusionModel/celeba/'             # turn on mixed precision
)

trainer.train()

