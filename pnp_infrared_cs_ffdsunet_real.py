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
from utils.utils import psnr, ssim, clip
from utils.ani import save_ani,weights_init_kaiming,HyperDatasetTest2
from model.model_sunet.models_FFDSUNet import FFDSUNet
from torch.utils.data import DataLoader
import yaml
import math
import torch.nn.functional as F




with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']


parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
args = parser.parse_args()
device_num = args.device
device = 'cuda:{}'.format(device_num)
torch.no_grad()
model = FFDSUNet(num_input_channels=1,opt=opt).to(device)
model.apply(weights_init_kaiming)

model = nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(torch.load('pretrained_models/net_FFDSUNet3.pth'))

model.eval()



test_set = HyperDatasetTest2(mode='test') ########### load test image
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


block_size=32
CS_ratio=0.10
bbb=block_size*block_size
sample=int(np.round(CS_ratio*1024))


################
D_data=sio.loadmat('test_data/real_scene/D.mat')
D_data = D_data['D'][:1000, :]
D_data = D_data.astype(np.float32)
phi_data =D_data

y_mat = sio.loadmat('test_data/real_scene/Y-X.mat')
Y = y_mat['Y'][:,190:191]###############182:183#4.127#############190:191#4.265

def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)

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
model_img=32
stride=32
def ffdsunet_denosing(x, sigma, flag):

    input_=x.unsqueeze(0)
    input_=input_.unsqueeze(0)
    # input_=input_.repeat(1,3,1,1)
    if flag:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        input_ = x.unsqueeze(0)
        input_ = input_.unsqueeze(0)
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    with torch.no_grad():
        # pad to multiple of 256
        square_input_, mask, max_wh = overlapped_square(input_.cuda(), kernel=model_img, stride=stride)
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
        for i, data in enumerate(square_input_):
            restored = model(square_input_[i],sigma.view(1, 1, 1, 1))#############1,1,32,32

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
    restored = x.unsqueeze(2) - restored[0]  ########pay attention to the output of network
    return restored

def ffdnet_denosing(x, sigma, flag):
    image_m, image_n = x.shape
    if flag:
        x_min = x.min().item()
        x_max = x.max().item()
        scale = 0.7
        shift = (1 - scale) / 2
        x = (x - x_min) / (x_max - x_min)
        x = x * scale + shift
        sigma = torch.tensor(sigma / (x_max - x_min) * scale, device=device)
    else:
        sigma = torch.tensor(sigma, device=device)

    frame_list = []
    with torch.no_grad():
        # for j in range(image_c):
        temp_x = x[:, :].view(1, 1, image_m, image_n)################1,1,32,32
            # estimate_img = model(temp_x, sigma.view(1, 1, 1, 1))

        pred_noise = model(temp_x, sigma.view(1, 1, 1, 1))

        estimate_img = temp_x - pred_noise  ########pay attention to the output of network
        frame_list.append(estimate_img[0, 0, :, :])
        x = torch.stack(frame_list, dim=2)

    if flag:
        x = (x - shift) / scale
        x = x * (x_max - x_min) + x_min
    return x

def run():



    Rec_ffdnet = np.zeros([64,48])
    Rec_ffdnet = torch.from_numpy(Rec_ffdnet)


    pred_time = []
    indices = {}
    i = 0
    count = 0
    tic = time.perf_counter()

    sigma_ = 50 / 255
    flag = True
    phi_gt = torch.from_numpy(phi_data).to(device)

    Rec_ffdnet = Rec_ffdnet.to(device)


    y = torch.from_numpy(Y).to(device)
    y1 = torch.zeros_like(y, dtype=torch.float32, device=device)

    phi_inv = phi_gt.T
    input = torch.mm(phi_inv, y)

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
        yb = yb / max(yb)  ###################for real
        # no Acceleration
        # temp = (y - yb) / (mask_sum)
        # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
        y1 = y1 + (y - yb)

        mask_inv=torch.linalg.inv(mask_sum)

        temp =torch.mm(mask_inv,(y1 - yb))

        x = x + torch.mm(mask.T,temp)
        x = x / max(x)  #####################for real
        net_input = x.view(64, 48)
        net_output = ffdsunet_denosing(net_input, sigma, flag).clamp(0, 1)###############32,32,1
        x = net_output.view(48 * 64, 1)
    Rec_ffdnet = x.view(64, 48)

    Rec_ffdnet = Rec_ffdnet.cpu().numpy()

    Rec_ffdnet = Rec_ffdnet * 255.0
    cv2.imwrite("test_out/ffdsunet/real/Rec_X_4265_real.bmp", Rec_ffdnet)

    print(time.time() - t)
    pred_time.append(time.time() - t)
    count += 1

    toc = time.perf_counter()
    print(sum(pred_time))


    psnr_ffdnet = 0
    ssim_ffdnet = 0
    x.clamp_(0, 1)


    return psnr_ffdnet, ssim_ffdnet

begin_time = time.time()
psnr_res, ssim_res = run()
end_time = time.time()
runing_time = end_time - begin_time
print('{:.2f}, {:.4f}, {:.2f}'.format(psnr_res, ssim_res, runing_time))