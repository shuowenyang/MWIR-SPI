import torch
import torch.optim as optim
import  torch.nn as nn
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import cvxpy as cp
import time
import argparse
import cv2
from skimage.measure import compare_psnr, compare_ssim
from utils.utils import psnr, ssim, clip
from utils.ani import save_ani,weights_init_kaiming,HyperDatasetTest2
from collections import OrderedDict
from model.SUNet import SUNet_model
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from model.TV_denoising import TV_denoising3d, TV_denoising
import yaml
import math
import torch.nn.functional as F

with open('training_sunet.yaml', 'r') as config:
    opt = yaml.safe_load(config)

parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
# parser.add_argument('--level', default=0)
args = parser.parse_args()
device_num = args.device
# level = float(args.level)
device = 'cuda:{}'.format(device_num)
torch.no_grad()

parser.add_argument('--weights',
                    default='/media/omnisky/c91e9985-5113-463d-83a6-6ec3405ef3a7/ysw/SUNet-main/checkpoints/Denoising/models/model_bestPSNR_outchannel1.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


# Load corresponding model architecture and weights
model = SUNet_model(opt)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()



test_set = HyperDatasetTest2(mode='test') ########### load test image
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


block_size=32
CS_ratio=0.50
bbb=block_size*block_size
sample=int(np.round(CS_ratio*1024))

# phi_data = sio.loadmat('phi_0_25_1089.mat')
#
# phi_data = phi_data['phi'][:sample, :bbb]  #########0.25-256,0.01-10,0.10-102
# phi_data = phi_data.astype(np.float32)



Phi_name = 'phi_sampling_50_109x1089.npy' #########50,109,1089
Phi_data = np.load(Phi_name)
Phi_data = Phi_data.astype(np.float32)
phi = torch.from_numpy(Phi_data)
phi=phi.view(5450,1089)
phi_data = phi[:sample, :1024]############0.50-512,0.40-410,0.30-307,0.20-205,0.10-102



def overlapped_square(timg, kernel=32, stride=32):
    patch_images = []
    b, c, h, w = timg.size()
    # 321, 481
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 1, X, X).type_as(timg)  # 3, h, w######################1/3
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)  # B, C, #patches, K, K
    patch = patch.permute(2, 0, 1, 4, 3)  # patches, B, C, K, K

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X


def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)

model_img=32
stride=32
def sunet_denosing(x, sigma, flag):

    input_=x.unsqueeze(0)
    input_=input_.unsqueeze(0)
    # input_=input_.repeat(1,3,1,1)

    with torch.no_grad():
        # pad to multiple of 256
        square_input_, mask, max_wh = overlapped_square(input_.cuda(), kernel=model_img, stride=stride)
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
        for i, data in enumerate(square_input_):
            restored = model(square_input_[i])#############1,1,32,32

            if i == 0:
                output_patch += restored
            else:
                output_patch = torch.cat([output_patch, restored], dim=0)

        B, C, PH, PW = output_patch.shape
        weight = torch.ones(B, C, PH, PH).type_as(output_patch)  # weight_mask

        patch = output_patch.contiguous().view(B, C, -1, model_img * model_img)
        patch = patch.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        patch = patch.contiguous().view(1, C * model_img * model_img, -1)

        weight_mask = weight.contiguous().view(B, C, -1, model_img * model_img)
        weight_mask = weight_mask.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        weight_mask = weight_mask.contiguous().view(1, C * model_img * model_img, -1)

        restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        restored /= we_mk

        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)
        restored = torch.clamp(restored, 0, 1)

    restored = restored.permute(0, 2, 3, 1)

    return restored[0]

def run():

    for t, ms in enumerate(test_loader):
        #########


        row = ms.shape[2]
        col = ms.shape[3]

        if np.mod(row, block_size) == 0:
            row_pad = 0
        else:
            row_pad = block_size - np.mod(row, block_size)

        if np.mod(col, block_size) == 0:
            col_pad = 0
        else:
            col_pad = block_size - np.mod(col, block_size)
        row_new = row + row_pad
        col_new = col + col_pad

        max_rol = row_new - block_size
        max_col = col_new - block_size
        ms = ms[:, :1, :max_rol, 0:max_col]

        Rec_ffdnet = np.zeros(np.shape(ms))
        Rec_ffdnet = torch.from_numpy(Rec_ffdnet)

        Rec_tv = np.zeros(np.shape(ms))
        Rec_tv= torch.from_numpy(Rec_tv)

        Rec_x = np.zeros(np.shape(ms))
        Rec_x = torch.from_numpy(Rec_x)


        pred_time = []
        indices = {}
        i = 0
        count = 0
        tic = time.perf_counter()
        while i + block_size <= max_rol:
            j = 0
            while j + block_size <= max_col:

                sigma_ = 50 / 255
                flag = True
                phi_gt = phi_data.to(device)
                img = ms[:, :, i:i + block_size, j:j + block_size].to(device)
                Rec_ffdnet = Rec_ffdnet.to(device)

                img = img.reshape(1, block_size * block_size)
                y = torch.mm(phi_gt, img.T).to(device)  # Performs a matrix-vector product
                y1 = torch.zeros_like(y, dtype=torch.float32, device=device)

                phi_inv = phi_gt.T
                input = torch.mm(phi_inv, y)
                # x = input.view(block_size, block_size)  #########32，32
                x = input
                t = time.time()
                mask = phi_gt
                # x=input
                mask_sum = torch.mm(mask, mask.T)
                mask_sum[mask_sum == 0] = 1

                for t in tqdm(range(30)):
                    if t == 20: flag = False
                    if t < 20:
                        sigma = 50. / 255
                    elif t < 30:
                        sigma = 25. / 255
                    elif t < 40:
                        sigma = 12. / 255
                    else:
                        sigma = 6. / 255

                    yb = torch.mm(mask, x)
                    # no Acceleration
                    # temp = (y - yb) / (mask_sum)
                    # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
                    y1 = y1 + (y - yb)

                    mask_inv=torch.linalg.inv(mask_sum)

                    # temp = (y1 - yb) / mask_sum
                    temp =torch.mm(mask_inv,(y1 - yb))
                    # x = x + 1 * (temp * mask.T)
                    x = x + torch.mm(mask.T,temp)########

                    net_input=x.view(block_size, block_size)#############32,32

                    net_output = sunet_denosing(net_input, sigma, flag).clamp(0, 1)

                    x=net_output.view(block_size*block_size,1)
                    # x = net_output[:,:,:1].view(block_size * block_size, 1)
                Rec_ffdnet[:, :, i:i + block_size, j:j + block_size] = x.view(block_size, block_size)

                print(time.time() - t)
                pred_time.append(time.time() - t)
                count += 1



                j += block_size
            i += block_size


            toc = time.perf_counter()
            print(sum(pred_time))
        Rec_ffdnet,  ms = Rec_ffdnet.squeeze(0).cpu().numpy().transpose(1, 2, 0), ms.squeeze(
            0).cpu().numpy().transpose(1, 2, 0)
        result1 = Rec_ffdnet[:ms.shape[0], :ms.shape[1], :]

        psnr_ffdnet=psnr(ms,result1)
        ssim_ffdnet=ssim(ms,result1)


        # plt.subplot(231)
        # plt.imshow(ms)
        # plt.title('Original Image')
        #
        # plt.subplot(232)
        # plt.imshow(Rec_ffdnet)
        # plt.title('ffdnet_re Image')
        #
        # plt.show()

        x.clamp_(0, 1)

        # Savemat

        ###############save reconstructed image

        cv2.imwrite("test_out/sunet/building/orig_%.1f.bmp"% (CS_ratio),ms)
        Rec_ffdnet = Rec_ffdnet * 255.0
        cv2.imwrite("test_out/sunet/building/Rec_%.1f.bmp"% (CS_ratio),Rec_ffdnet)

    return psnr_ffdnet, ssim_ffdnet

begin_time = time.time()

psnr_res, ssim_res = run()
end_time = time.time()
runing_time = end_time - begin_time
print('{:.2f}, {:.4f}, {:.2f}'.format(psnr_res, ssim_res, runing_time))