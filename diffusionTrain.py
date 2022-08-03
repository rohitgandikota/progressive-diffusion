#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:03:26 2022

@author: Rohit Gandikota
"""
from progressiveDiffusion import initProgression, Trainer

import torch

torch.cuda.empty_cache()
model = initProgression(
    dim = 128,
    dim_mults = (1, 2, 4, 8)
).cuda()



trainer = Trainer(
    model,
    'D:\\Deep Learning\\Data\\tiny-imagenet-200\\tiny-imagenet-200\\train\\n01629819\\images',
    image_size= 128,
    train_batch_size = 1,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True
)

trainer.train()
