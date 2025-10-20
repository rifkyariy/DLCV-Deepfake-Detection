import os
import torch
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from functools import partial
import multiprocessing

# Assuming these imports are from your project structure
from model_paper import Detector
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# --- Global variables for worker processes ---
# These will be initialized once per worker to avoid passing large models
model = None
face_detector = None
device = None

def init_worker(weight_name, dev_str):
    """
    Initializer function for each worker in the multiprocessing pool.
    Loads models into memory for that specific process.
    """
    global model, face_detector, device
    device = torch.device(dev_str)

    # Load the main detector model
    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(weight_name, map_location=device)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    # Load the RetinaFace face detector model
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    # print(f"Worker process {os.getpid()} initialized.") # Optional: uncomment for debugging


def process_video(filename, n_frames):
    """
    Processes a single video file. This function is executed by each worker process.
    """
    global model, face_detector, device
    try:
        # 1. Extract faces and their corresponding frame indices
        face_list, idx_list = extract_frames(filename, n_frames, face_detector)

        # If no faces are found, return a neutral score
        if not face_list:
            return 0.5

        with torch.no_grad():
            # 2. Run inference on the batch of faces
            img_tensor = torch.tensor(face_list).to(device).float() / 255
            preds = model(img_tensor).softmax(1)[:, 1]
            
            # 3. Vectorized post-processing on GPU
            idx_tensor = torch.tensor(idx_list, device=device)
            
            # Find the max prediction for each unique frame index.
            # This is much faster than the original Python loops.
            unique_indices = torch.unique(idx_tensor)
            max_preds = [torch.max(preds[idx_tensor == i]) for i in unique_indices]
            
            # Calculate the final score as the mean of the max predictions
            final_pred = torch.stack(max_preds).mean().item()

    except Exception as e:
        # print(f"Error processing {os.path.basename(filename)}: {e}")
        final_pred = 0.5  # Return neutral score on error
        
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
    # Use a user-defined number of workers to avoid overwhelming the GPU
    num_workers = args.num_workers
    print(f"Starting evaluation with {num_workers} parallel workers on device '{args.device}'...")
    
    # Use functools.partial to "fix" the n_frames argument for the process_video function
    task_func = partial(process_video, n_frames=args.n_frames)

    # Set up the initializer function to pass arguments to init_worker for each process
    initializer = partial(init_worker, args.weight_name, args.device)

    output_list = []
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers, initializer=initializer) as pool:
        # Use imap to process the video list and tqdm for a progress bar
        # imap is memory-efficient for large lists
        for result in tqdm(pool.imap(task_func, video_list), total=len(video_list)):
            output_list.append(result)

    # --- Calculate and Print Final Score ---
    auc = roc_auc_score(target_list, output_list)
    print(f'\n{args.dataset} | AUC: {auc:.4f}')


if __name__ == '__main__':
    # --- Seeding and Setup ---
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fast, parallelized deepfake detection evaluation.")
    parser.add_argument('-w', dest='weight_name', type=str, required=True, help="Path to the trained model weights (.tar file)")
    parser.add_argument('-d', dest='dataset', type=str, required=True, help="Name of the dataset to evaluate (e.g., FF, DFD)")
    parser.add_argument('-n', dest='n_frames', default=32, type=int, help="Number of frames to extract per video")
    parser.add_argument('--num-workers', dest='num_workers', default=4, type=int, help="Number of parallel worker processes to spawn")
    parser.add_argument('--device', dest='device', default='cuda', type=str, help="Device to use for inference (e.g., 'cuda', 'cuda:0', 'cpu')")
    args = parser.parse_args()

    # Set start method to 'spawn' for CUDA compatibility in multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    main(args)

