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
from model.network_ADNet import ADNet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from model.TV_denoising import TV_denoising3d, TV_denoising

parser = argparse.ArgumentParser(description='Select device')
parser.add_argument('--device', default=0)
# parser.add_argument('--level', default=0)
args = parser.parse_args()
device_num = args.device
# level = float(args.level)
device = 'cuda:{}'.format(device_num)
torch.no_grad()
# model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)
torch.no_grad()
net = ADNet(channels=1, num_of_layers=17)
device_ids = [0]
model = nn.DataParallel(net, device_ids=device_ids).cuda()
model.load_state_dict(torch.load('pretrained_models/ADNet_70.pth'))
model.eval()


################
D_data=sio.loadmat('test_data/real_scene/D.mat')
D_data = D_data['D'][:1000, :]
D_data = D_data.astype(np.float32)
phi_data =D_data

y_mat = sio.loadmat('test_data/real_scene/Y-O.mat')
Y = y_mat['Y'][:,182:183]


def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)

def adnet_denosing(x):
    image_m, image_n, image_c = x.shape
    # noise
    torch.manual_seed(0)  # set the seed
    # noise = torch.FloatTensor(x.size()).normal_(mean=0, std=25 / 255.).to(device)
    # noisy image
    # INoisy = x + noise
    # ISource = Variable(x)
    # INoisy = Variable(INoisy)
    # ISource = ISource.cuda()
    INoisy = x.cuda()
    frame_list = []

    with torch.no_grad():  # this can save much memory
        for j in range(image_c):
            temp_x = INoisy[:, :, j].view(1, 1, image_m, image_n)
            estimate_img = torch.clamp(model(temp_x), 0., 1.)
            frame_list.append(estimate_img[0, 0, :, :])
        Out = torch.stack(frame_list, dim=2)
    return Out


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
        yb = yb / max(yb)  ###################for real
        # no Acceleration
        # temp = (y - yb) / (mask_sum)
        # x = x + 1 * (temp.unsqueeze(2).expand_as(mask) * mask)
        y1 = y1 + (y - yb)

        mask_inv=torch.linalg.inv(mask_sum)

        # temp = (y1 - yb) / mask_sum
        temp =torch.mm(mask_inv,(y1 - yb))
        # x = x + 1 * (temp * mask.T)
        x = x + torch.mm(mask.T,temp)
        x = x / max(x)  #####################for real

        net_input = x.view(64, 48)
        net_input=net_input.unsqueeze(2)

        net_output = adnet_denosing(net_input).clamp(0, 1)

        x = net_output.view(48 * 64, 1)

    Rec_ffdnet = x.view(64, 48)
    Rec_ffdnet= Rec_ffdnet.cpu().numpy()
    Rec_ffdnet = Rec_ffdnet * 255.0
    cv2.imwrite("test_out/adnet/real/Rec_O_real.bmp", Rec_ffdnet)

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