import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import datetime
import math
import torch.nn as nn
import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import scipy.io

from PIL import Image



# now_date = str(datetime.datetime.now())
fig = plt.figure()
# plt.title(now_date)
ims = []
def animation_generate(img):
    ims_i = []
    im = plt.imshow(img, cmap='gray')
    ims_i.append([im])
    return ims_i

def save_ani(x_list, filename='v.gif', fps=60):
    ims = []
    fig = plt.figure()
    for img in x_list:
        ims += animation_generate(img)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, fps=fps, writer='ffmpeg')#'imagemagick')

def weights_init_kaiming(lyr):
	r"""Initializes weights of the model according to the "He" initialization
	method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
	This function is to be called by the torch.nn.Module.apply() method,
	which applies weights_init_kaiming() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
			clamp_(-0.025, 0.025)
		nn.init.constant(lyr.bias.data, 0.0)



def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


# test 512*512 images
class CAVEDatasetTest(udata.Dataset):
    def __init__(self, mode='test'):
        if mode != 'test':
            raise Exception("Invalid mode!", mode)
        data_path = '/work/work_ysw/SSPSR/DeepInverse-Pytorch/Harvard/Hyper_Valid_Clean'
        data_names = glob.glob(os.path.join(data_path, '*.mat'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # mat = h5py.File(self.keys[index], 'r')
        mat = scipy.io.loadmat(self.keys[index])
        hyper = np.float32(np.array(mat['ref']))

        hyper = normalize(hyper, max_val=1., min_val=0.)


        return hyper

# test 640*512 images
class HyperDatasetTest2(udata.Dataset):
    def __init__(self, mode='test'):
        if mode != 'test':
            raise Exception("Invalid mode!", mode)
        data_path = 'test_data'
        data_names = glob.glob(os.path.join(data_path, 'chimney.bmp'))

        self.keys = data_names
        random.shuffle(self.keys)
        # self.keys.sort()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # mat = h5py.File(self.keys[index], 'r')
        # OD_img = cv2.imread(self.OD_img_list[index])######BGR
        hyper = Image.open(self.keys[index])
        hyper = np.array(hyper, dtype=np.uint8)
        # mat = scipy.io.loadmat(self.keys[index])
        # hyper = np.float32(np.array(mat['image']))
        # hyper = np.transpose(hyper, [2, 1, 0])
        # hyper = normalize(hyper, max_val=1., min_val=0.)
        hyper=hyper/255
        hyper = torch.Tensor(hyper)
        # hyper=hyper.unsqueeze(dim=0)
        hyper=hyper[:,:,0].unsqueeze(dim=0)##########1,128,128

        return hyper



if __name__ == '__main__':
    ani = animation.ArtistAnimation(fig, ims)
    ani.save("v.gif", writer='imagemagick')
