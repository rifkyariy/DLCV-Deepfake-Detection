import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector5 as Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import pywt
import warnings
import cv2
from skimage import feature
from skimage import filters
warnings.filterwarnings('ignore')

def main(args, model_path, w):
    # --- MODEL SETUP ---
    model = Detector()
    model = model.to(device)
    
    # Load weights safely
    try:
        cnn_sd = torch.load(model_path, map_location=torch.device('cpu'))["model"]
        model.load_state_dict(cnn_sd)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    # Face Detector Setup
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    # --- DATASET INIT ---
    if args.dataset == 'FFIW':
        video_list, target_list = init_ffiw()
    elif args.dataset == 'FF':
        video_list, target_list = init_ff_t(args.type)
    elif args.dataset == 'DFD':
        video_list, target_list = init_dfd()
    elif args.dataset == 'DFDC':
        video_list, target_list = init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list, target_list = init_dfdcp()
    elif args.dataset == 'CDF':
        video_list, target_list = init_cdf()
    else:
        raise NotImplementedError("Dataset not found")

    output_list = []
    
    # Main Loop
    for filename in tqdm(video_list):
        try:
            # 1. Extract Frames (This sits in CPU RAM, but usually is not too huge)
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)
            
            if len(face_list) == 0:
                # Handle cases where no faces are detected
                output_list.append(0.5) 
                continue

            gpu_tensor_batch = []

            # 2. Process frames ONE BY ONE and move to GPU immediately
            # This prevents RAM spikes
            with torch.no_grad():
                for f in range(len(face_list)):
                    # Ensure float32 to save memory (default is often float64)
                    face = face_list[f].astype('float32') / 255.0
                    facee = np.transpose(face.copy(), (1, 2, 0))

                    b, g, r = cv2.split(facee)

                    # DWT (CPU Bound)
                    cA_r, _ = pywt.dwt2(r, 'sym2', mode='reflect')
                    cA_g, _ = pywt.dwt2(g, 'sym2', mode='reflect')
                    cA_b, _ = pywt.dwt2(b, 'sym2', mode='reflect')

                    # Resize
                    cA_r = cv2.resize(cA_r, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')
                    cA_g = cv2.resize(cA_g, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')
                    cA_b = cv2.resize(cA_b, (380, 380), interpolation=cv2.INTER_LINEAR).astype('float32')

                    # Average
                    cA_r = (cA_r + r) / 2
                    cA_g = (cA_g + g) / 2
                    cA_b = (cA_b + b) / 2

                    # Stack numpy array
                    img_dwt = np.array([cA_r, cA_g, cA_b])

                    # --- CRITICAL FIX START --- 
                    # Convert to tensor and send to GPU individually.
                    # unsqueeze(0) adds the batch dimension: [3, 380, 380] -> [1, 3, 380, 380]
                    tensor_frame = torch.from_numpy(img_dwt).float().unsqueeze(0).to(device)
                    gpu_tensor_batch.append(tensor_frame)
                    # --- CRITICAL FIX END ---

                # 3. Create the batch directly on GPU
                if len(gpu_tensor_batch) > 0:
                    img_batch = torch.cat(gpu_tensor_batch, dim=0)
                    
                    # Inference
                    pred = model(img_batch).softmax(1)[:, 1]
                    
                    # Cleanup VRAM immediately
                    del img_batch
                    del gpu_tensor_batch
                else:
                    pred = torch.tensor([])

            # 4. Post-processing prediction logic
            pred_list = []
            idx_img = -1
            
            # Move pred to CPU for list operations to avoid GPU synchronization overhead
            pred_cpu = pred.cpu().numpy()
            
            for i in range(len(pred_cpu)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred_cpu[i].item())
            
            pred_res = np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i] = max(pred_list[i])
            
            final_pred = pred_res.mean()
            output_list.append(final_pred)
            
            # Explicit cache clear to keep VRAM healthy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Skipping video {filename} due to error: {e}")
            output_list.append(0.5)
            continue

    # Final Metric Calculation
    try:
        auc = roc_auc_score(target_list, output_list)
        print(f'{args.dataset}| AUC: {auc:.4f}')
    except Exception as e:
        print(f"Error calculating AUC: {e}")

if __name__=='__main__':
    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str, required=True)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-t',dest='type',default="Face2Face",type=str)
    args=parser.parse_args()
    
    model_path = args.weight_name
    w = os.path.basename(model_path)
    
    print(f"Processing: {w}")
    main(args, model_path, w)