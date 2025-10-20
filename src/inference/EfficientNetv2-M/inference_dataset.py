import os
import torch
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from functools import partial
import multiprocessing
import sys

# Add project's 'src' directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, src_dir)

from model import Detector 
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- Global variables for worker processes ---
model = None
face_detector = None
device = None

def preload_models():
    """
    Initializes models once in the main process to download and cache weights.
    This prevents race conditions and redundant downloads by worker processes.
    """
    print("Pre-loading models to cache...")
    # Load RetinaFace and EfficientNetV2 weights into the cache on the CPU.
    # We only need to trigger the download, not use the models here.
    get_model("resnet50_2020-07-20", max_size=2048, device='cpu')
    Detector()
    print("Models are cached.")
# --------------------------------------

def init_worker(weight_name, dev_str):
    """
    Initializer for each worker. Now loads models from the pre-warmed cache.
    """
    global model, face_detector, device
    device = torch.device(dev_str)

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(weight_name, map_location=device)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

def process_video(filename, n_frames):
    """
    Processes a single video file.
    """
    global model, face_detector, device
    try:
        face_list, idx_list = extract_frames(filename, n_frames, face_detector)

        if not face_list:
            return 0.5

        with torch.no_grad():
            img_tensor = torch.tensor(face_list).to(device).float() / 255
            preds = model(img_tensor).softmax(1)[:, 1]
            
            idx_tensor = torch.tensor(idx_list, device=device)
            unique_indices = torch.unique(idx_tensor)
            max_preds = [torch.max(preds[idx_tensor == i]) for i in unique_indices]
            final_pred = torch.stack(max_preds).mean().item()

    except Exception as e:
        final_pred = 0.5
        
    return final_pred

def main(args):
    # --- Dataset Initialization ---
    if args.dataset == 'FFIW':
        video_list, target_list = init_ffiw()
    elif args.dataset == 'FF':
        video_list, target_list = init_ff()
    elif args.dataset == 'DFD':
        video_list, target_list = init_dfd()
    elif args.dataset == 'DFDC':
        video_list, target_list = init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list, target_list = init_dfdcp()
    elif args.dataset == 'CDF':
        video_list, target_list = init_cdf()
    else:
        raise NotImplementedError

    # --- Parallel Processing Setup ---
    num_workers = args.num_workers
    print(f"Starting evaluation with {num_workers} parallel workers on device '{args.device}'...")
    
    task_func = partial(process_video, n_frames=args.n_frames)
    initializer = partial(init_worker, args.weight_name, args.device)

    output_list = []
    with multiprocessing.Pool(processes=num_workers, initializer=initializer) as pool:
        for result in tqdm(pool.imap(task_func, video_list), total=len(video_list)):
            output_list.append(result)

    auc = roc_auc_score(target_list, output_list)
    print(f'\n{args.dataset} | AUC: {auc:.4f}')

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Fast, parallelized deepfake detection evaluation.")
    parser.add_argument('-w', dest='weight_name', type=str, required=True, help="Path to the trained model weights (.tar file)")
    parser.add_argument('-d', dest='dataset', type=str, required=True, help="Name of the dataset to evaluate (e.g., FF, DFD)")
    parser.add_argument('-n', dest='n_frames', default=32, type=int, help="Number of frames to extract per video")
    parser.add_argument('--num-workers', dest='num_workers', default=4, type=int, help="Number of parallel worker processes to spawn")
    parser.add_argument('--device', dest='device', default='cuda', type=str, help="Device to use for inference (e.g., 'cuda', 'cuda:0', 'cpu')")
    args = parser.parse_args()

    # Pre-load models in the main process to ensure weights are downloaded and cached.
    # This must be done BEFORE the multiprocessing pool is created.
    preload_models()

    multiprocessing.set_start_method('spawn', force=True)
    main(args)