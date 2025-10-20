import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main(args):

    # Initialize detector (MobileViT backbone now). Attempt robust checkpoint loading.
    model = Detector()
    if args.weight_name:
        try:
            ckpt = torch.load(args.weight_name, map_location=device)
            # Accept formats: {'model': state_dict}, {'state_dict':...}, or raw state dict
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            # Strip 'module.' if present
            new_sd = {}
            for k,v in state_dict.items():
                nk = k[7:] if k.startswith('module.') else k
                nk = nk.replace('head.', 'classifier.')
                if nk.startswith('classifier') and v.dim()==2 and v.shape[0] != 2:
                    # skip classifier mismatch
                    continue
                new_sd[nk] = v
            missing, unexpected = model.load_state_dict(new_sd, strict=False)
            print(f"Loaded weights from {args.weight_name} (missing={len(missing)} unexpected={len(unexpected)})")
        except Exception as e:
            print(f"Failed to load checkpoint {args.weight_name}: {e}")
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff()
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    for filename in tqdm(video_list):
        try:
            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

            with torch.no_grad():
                img=torch.tensor(face_list).to(device).float()/255
                pred=model(img).softmax(1)[:,1]
                
                
            pred_list=[]
            idx_img=-1
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')






if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)

