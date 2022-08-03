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
from progressively_growing_diffusion import Unet, GaussianDiffusion
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
torch.cuda.empty_cache()

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid



def inferenceDDPM(folder_result, folder_model,folder_dataset,num_of_samples):
    models = glob.glob(folder_model + '/*.pt')
    for model_file in models:
        # Selecting the model checkpoints that are multiple of 25
        if int(os.path.basename(model_file).split('.')[0].split('-')[-1])%25 !=0:
            continue
        else:
            print(f"Inference for Epoch Model {int(os.path.basename(model_file).split('.')[0].split('-')[-1])}")
        # Initialise the final model architecture for inference
        mod = Unet(dim = 128, dim_mults = (1, 2, 4, 8)).cuda()
        diff = GaussianDiffusion(mod,image_size = 128,timesteps = 1000, loss_type = 'l1').cuda()
        model_check = torch.load(model_file)
        model_dict = model_check['ema']
        diff.load_state_dict(model_dict)
        num_batches = num_of_samples//100 # Number of batches of size 100 to be considered for FID
        for i in range(num_batches):
            sampled_images = diff.sample(batch_size = 100)
            a = np.array(sampled_images.cpu())
            a = np.einsum('ijkl->iklj',a)
            if i == 0 :
                a_s = a
            else:
                a_s = np.vstack([a_s,a])
            del(sampled_images)
            del a
            torch.cuda.empty_cache()
        np.savez_compressed(f'{folder_result}/{os.path.basename(model_file).split(".")[0]}_gen.npz', a_s)
    
    # Clear cache from GPU
    try:
        del diff
    except:
        pass
    try:
        del a_s
    except:
        pass
    try:
        del mod
    except:
        pass
    try:
        del model_check
    except:
        pass
    try:
        del model_dict
    except:
        pass
    torch.cuda.empty_cache()
    
    vars_ = glob.glob(f'{folder_result}/*.npz')
    FID=[]
    for i in range(0,len(vars_)*25+1,25):
        # Selecting only the saved data from modelcheckpoints that are multiples of 25
        a = [var for var in vars_ if str(i) in var][0]
        # Loading the data
        np_gen_images = np.load(a)['arr_0']
        # Save samples of individually generated images
        if i == len(vars_)*25:  
            for i in range(25):
                fig=plt.figure(figsize=(2,2))
                plt.imshow(np_gen_images[i])
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{folder_result}/EMASample-{i}_{os.path.basename(model_file)}.png')
        # Path to folder containing the corresponding training data
        
        train_images = glob.glob(folder_dataset+'/*.png')
        train_images.extend(glob.glob(folder_dataset+'/*.jpg'))
        train_images.extend(glob.glob(folder_dataset+'/*.jpeg'))
        # Loading the actual training dataset to compare for FID (For precise calculation, load the entire dataset and generate as many samples)
        actual_images = np.random.choice(train_images,len(np_gen_images))
        np_act_images= []
        for im in actual_images:
            np_act_images.append(plt.imread(im))
        np_act_images = np.array(np_act_images)
        # Reshape for smaller resolution datasets like cifar10 as InceptionV3 has minimum input requirement of 75
        if np_act_images.shape[2]<128:
            np_act_images = scale_images(np_act_images, (75,75,3))
            np_gen_images = scale_images(np_gen_images, (75,75,3))
        if np_act_images.shape[2] != np_gen_images.shape[2]:
            np_act_images = scale_images(np_act_images, (np_gen_images.shape[2],np_gen_images.shape[2],3))
        np_act_images = preprocess_input(np_act_images)
        np_gen_images = preprocess_input(np_gen_images)
        
        # fid between images1 and images1
        incep_model = InceptionV3(include_top=False, pooling='avg', input_shape=(np_gen_images.shape[2],np_gen_images.shape[2],3))
        fid = calculate_fid(incep_model, np_gen_images, np_act_images)
        print('FID : %.3f' % fid)
        FID.append(fid)               
        np.savetxt(folder_result+'/FID.csv', FID, delimiter=',')
    
    fig=plt.figure(figsize=(5,5))
    plt.rcParams['figure.facecolor'] = 'white'
    t = np.arange(0,len(FID))
    plt.title('FID Trend in Pro-DDPM')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.plot(t, FID)
    plt.savefig(folder_result+'/FID.png')
    
if __name__=='__main__':
    
    folder_result = '/data/results/animal-hq/cat/'
    folder_model = '/data/models/animal-hq/cat/'
    folder_dataset = '/data/dataset/animal-hq/train/cat/'
    num_of_samples = 5000
    inferenceDDPM(folder_result, folder_model,folder_dataset,num_of_samples)