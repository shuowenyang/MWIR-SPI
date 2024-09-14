"""
Definition of the FFDNet model and its custom layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch.nn as nn
from torch.autograd import Variable
import functions
from model.model_sunet.SUNet_detail import SUNet
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x


class UpSampleFeatures(nn.Module):
	r"""Implements the last layer of FFDNet
	"""
	def __init__(self):
		super(UpSampleFeatures, self).__init__()
	def forward(self, x):
		return functions.upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
	r"""Implements the middel part of the FFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):
		super(IntermediateDnCNN, self).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = input_features
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			raise Exception('Invalid number of input features')

		layers = []
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)
	def forward(self, x):
		out = self.itermediate_dncnn(x)
		return out

class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=2,
                               out_chans=4,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],
                               num_heads=config['SWINUNET']['HEAD_NUM'],
                               window_size=config['SWINUNET']['WIN_SIZE'],
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],
                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
class FFDSUNet(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels,opt):
		super(FFDSUNet, self).__init__()
		self.num_input_channels = num_input_channels
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')
		self.config = opt
		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()
		# self.upsamplefeatures = PatchExpand(input_resolution=(128,128),dim=4, dim_scale=2, norm_layer=nn.LayerNorm)
		self.intermediate_sunet= SUNet_model(self.config)

		self.conv_last = nn.Conv2d(4, 1, 3, 1, 1)


	def forward(self, x, noise_sigma):############x:128,1,44,44;nosise:128
		concat_noise_x = functions.concatenate_input_noise_map(\
				x.data, noise_sigma.data)
		concat_noise_x = Variable(concat_noise_x)############concat:8,1,32,32

		logits = self.intermediate_sunet(concat_noise_x)#####B,4,32,32

		#h_dncnn = self.intermediate_dncnn(concat_noise_x)#########128,4,22,22
		# pred_noise = self.upsamplefeatures(logits)######33128,1,44,44
		pred_noise=self.conv_last(logits)
		return pred_noise
