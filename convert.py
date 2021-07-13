import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import jit

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import save_model, load_model
from utils import yaml_config_hook
from simclr.modules import SingleViT as VViT
import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score
from datetime import datetime
from sklearn import metrics
from setLogger import *
import pandas as pd

parser = argparse.ArgumentParser(description="SimCLR")
config = yaml_config_hook("./config/config_single_test.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--f', type=bool, default=True)
args = parser.parse_args()
# args.model_path = os.path.join('checkpoint', "ViTRes50-16-gmlp-im256-MF")


model = VViT(0, args, mode='test')

model = model.to('cuda')

args.current_epoch = 0
# best_f1 = load_model(args, model, useBest=args.useBest)
# if isinstance(model, torch.nn.DataParallel):
#     model_dict = model.module.state_dict()
# else:
#     model_dict = model.state_dict()
    
# torch.save(model_dict, 'ViTRes50-16-gmlp-im256-MF.pth')
model.load_state_dict(torch.load('ViTRes50-16-gmlp-im256-MF.pth'))

data = {}
data['fn']={}
data['img'] = {}
for i in range(args.FRR):
    data['img'][i] = torch.ones(1, 3, 224, 224)
net_trace = jit.trace(model, data)
jit.save(net_trace, 'model.zip')