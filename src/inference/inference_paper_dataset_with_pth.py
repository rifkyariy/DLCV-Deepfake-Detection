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
from model_paper import Detector
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

    model=Detector()
    model=model.to(device)
    
    # Load model weights (handles both compressed and regular checkpoints)
    print(f"Loading model from: {args.weight_name}")
    if args.compressed:
        print("Loading compressed checkpoint...")
        missing_keys, unexpected_keys = model.load_compressed_state_dict(
            args.weight_name, 
            strict=False
        )
        if missing_keys:
            print(f"⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
        print("✓ Compressed model loaded successfully!")
    else:
        print("Loading regular checkpoint...")
        cnn_sd = torch.load(args.weight_name, map_location=device)
        model.load_state_dict(cnn_sd, strict=False)
        print("✓ Model loaded successfully!")
    
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
    parser.add_argument('-w',dest='weight_name',type=str, help='Path to model weights (.pth file)')
    parser.add_argument('-d',dest='dataset',type=str, help='Dataset name (FFIW, FF, DFD, DFDC, DFDCP, CDF)')
    parser.add_argument('-n',dest='n_frames',default=32,type=int, help='Number of frames to extract')
    parser.add_argument('--compressed', action='store_true', 
                        help='Use this flag if loading a compressed model (8-bit or 16-bit quantized)')
    args=parser.parse_args()

    main(args)