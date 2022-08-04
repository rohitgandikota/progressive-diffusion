#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:03:26 2022

@author: Rohit Gandikota
"""
from progressiveDiffusion import initProgression, Trainer, finalModel

import torch

torch.cuda.empty_cache()

model = initProgression(
    dim = 64,
    dim_mults = (1, 2, 4,8)
).cuda(1)


trainer = Trainer(
    model,
    folder = '/data/animal-hq/train/cat',
    image_size= 128,
    train_batch_size = 1,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    timesteps = 1000,                 # Train sampling steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,
    loss_type = 'l1',
    step_start_ema = 2000,
    update_ema_every = 10,
    save_and_sample_every = 1000,
    results_folder = './results'
)


trainer.train()


#%% Model Visualization
from torchsummary import summary
finalModel = finalModel(
    dim = 64,
    dim_mults = (1, 2, 4,8)
).cuda(1)

x = torch.randn(0,2).cuda(1)
summary(finalModel, (3, 64, 64),torch.randint(0, 2, (2,), device=x.device).long())

