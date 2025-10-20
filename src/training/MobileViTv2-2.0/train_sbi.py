import os
import sys

# --- Adjust sys.path to include 'src' directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time

# --- Local Imports ---
from utils.sbi import SBI_Dataset
from utils.logs import log
from utils.funcs import load_json
from model import Detector
import wandb
import multiprocessing

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

def compute_accuracy(pred, true):
    """Computes the accuracy of predictions without unnecessary log_softmax."""
    # .argmax() works directly on logits
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = (pred_idx == true.cpu().numpy())
    return sum(tmp) / len(pred_idx)

def setup_environment(seed=5):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # When set to True, this helps optimize performance but might affect reproducibility.
    # It's a good trade-off for speed in most cases.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# -----------------------------------------------------------
# Core Training and Validation Functions
# -----------------------------------------------------------

def train_one_epoch(model, loader, criterion, scaler, device, epoch, n_epoch, accumulation_steps, use_amp):
    """Handles the training loop for a single epoch with SAM optimizer."""
    model.train()
    train_loss = 0.
    train_acc = 0.
    
    # Empty cache before epoch to prevent fragmentation
    torch.cuda.empty_cache()
    
    # NOTE: The SAM process is not directly compatible with standard gradient accumulation.
    # This loop is designed to work correctly when `accumulation_steps` is 1.
    if accumulation_steps > 1:
        print("Warning: SAM is not compatible with gradient accumulation. "
              "For correct behavior, please use --accumulation-steps 1.")

    epoch_start = time.time()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")
    for step, data in enumerate(progress_bar):
        img = data['img'].to(device, non_blocking=True).float()
        target = data['label'].to(device, non_blocking=True).long()

        # --- SAM Step 1: Ascent Step ---
        with torch.cuda.amp.autocast(enabled=use_amp):
            output_1 = model(img)
            loss_1 = criterion(output_1, target)
        
        # First backward pass
        scaler.scale(loss_1).backward()
        
        # first_step perturbs the weights and zeros the gradients
        model.optimizer.first_step(zero_grad=True)

        # --- SAM Step 2: Descent Step ---
        with torch.cuda.amp.autocast(enabled=use_amp):
            output_2 = model(img)
            loss_2 = criterion(output_2, target)
        
        # Second backward pass
        scaler.scale(loss_2).backward()
        
        # Unscale gradients before the final optimizer step
        scaler.unscale_(model.optimizer)
        
        # second_step performs the actual weight update
        model.optimizer.second_step(zero_grad=True)

        # Update the scaler for the next iteration
        scaler.update()

        # Logging based on the first pass
        train_loss += loss_1.item()
        acc = compute_accuracy(output_1, target)
        train_acc += acc
        progress_bar.set_postfix(loss=f"{loss_1.item():.4f}", acc=f"{acc:.4f}")
        
        # Periodically empty cache to prevent memory fragmentation
        if step % 100 == 0 and step > 0:
            torch.cuda.empty_cache()

    train_time = time.time() - epoch_start
    avg_train_loss = train_loss / len(loader)
    avg_train_acc = train_acc / len(loader)
    return avg_train_loss, avg_train_acc, train_time

def validate_one_epoch(model, loader, criterion, device, epoch, n_epoch, use_amp):
    """Handles the validation loop for a single epoch."""
    model.eval() # Use eval() for validation
    val_loss = 0.
    val_acc = 0.
    output_dict = []
    target_dict = []
    
    # Empty cache and sync before validation
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    val_start = time.time()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")
    
    with torch.no_grad():
        for step, data in enumerate(progress_bar):
            if step == 0:
                first_batch_time = time.time() - val_start
                print(f"   First validation batch loaded in {first_batch_time:.2f}s")
                
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            
            # Use autocast for consistency/speed
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(img)
                loss = criterion(output, target)
            
            val_loss += loss.item()
            acc = compute_accuracy(output, target)
            val_acc += acc
            output_dict.extend(output.softmax(1)[:, 1].cpu().data.numpy().tolist())
            target_dict.extend(target.cpu().data.numpy().tolist())
            
            # Periodically empty cache during validation
            if step % 50 == 0 and step > 0:
                torch.cuda.empty_cache()

    val_time = time.time() - val_start
    avg_val_loss = val_loss / len(loader)
    avg_val_acc = val_acc / len(loader)
    val_auc = roc_auc_score(target_dict, output_dict)
    return avg_val_loss, avg_val_acc, val_auc, val_time

# -----------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------

def main(args):
    cfg = load_json(args.config)
    setup_environment()
    device = torch.device('cuda')

    # --- Weights & Biases Initialization ---
    wandb_run_id = None
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        wandb_run_id = checkpoint.get('wandb_run_id')
    else:
        checkpoint = None

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=cfg,
            id=wandb_run_id,
            resume="allow"
        )
        wandb.config.update(args, allow_val_change=True)

    # --- Dataset and DataLoader Configuration ---
    image_size = cfg['image_size']
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
    
    # SAM optimizer does 2 forward passes, so we need smaller batch size to fit in memory
    train_batch_size = batch_size // 2  # Halve for SAM's double forward pass
    val_batch_size = batch_size // 2    # Keep consistent with training
    
    train_dataset = SBI_Dataset(phase='train', image_size=image_size)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size)
   
    # Force at least 2 workers for better performance
    actual_num_workers = max(2, args.num_workers)
    use_pin_memory = actual_num_workers > 0
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Train batch size: {train_batch_size}")
    print(f"   Val batch size: {val_batch_size}")
    print(f"   Num workers: {actual_num_workers} (requested: {args.num_workers})")
    print(f"   Pin memory: {use_pin_memory}")
    print(f"   Use AMP: {args.use_amp}\n")
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=actual_num_workers,
                        pin_memory=use_pin_memory,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn,
                        persistent_workers=True,
                        prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=actual_num_workers,
                        pin_memory=use_pin_memory,
                        worker_init_fn=val_dataset.worker_init_fn,
                        persistent_workers=True,
                        prefetch_factor=2)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}\n")
    
    # --- Model, Optimizer, Scheduler, and Scaler Setup ---
    model = Detector(lr=cfg['lr']).to(device)
    
    if not args.no_wandb:
        wandb.watch(model, log='all', log_freq=100)
    
    # Initialize the GradScaler for Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # --- Checkpoint Resuming and Output Path Setup ---
    start_epoch = 0
    save_path = ''
    best_val_loss = float('inf')
    early_stop_counter = 0

    if args.resume and checkpoint:
        print(f"Loading checkpoint '{args.resume}'")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Check if the scaler state exists before loading it
        if 'scaler' in checkpoint:
            print("Loading GradScaler state from checkpoint.")
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            print("Scaler state not found in checkpoint. Initializing a new one.")
            
        save_path = checkpoint['save_path']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        print(f"Resumed training from epoch {start_epoch}")
    
    if not save_path:
        now = datetime.now()
        run_name = f"{args.session_name}_{os.path.splitext(os.path.basename(args.config))[0]}_{now.strftime('%m_%d_%H_%M_%S')}"
        save_path = os.path.join('output', run_name)
        os.makedirs(os.path.join(save_path, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)
        
    logger = log(path=os.path.join(save_path, "logs/"), file="losses.logs")
    criterion = nn.CrossEntropyLoss()
    
    # --- Main Training Loop ---
    for epoch in range(start_epoch, cfg['epoch']):
        # --- Training Phase ---
        avg_train_loss, avg_train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, scaler, device, 
            epoch, cfg['epoch'], args.accumulation_steps, args.use_amp
        )
        current_lr = model.optimizer.param_groups[0]['lr']
        log_text = f"Epoch {epoch+1}/{cfg['epoch']} | LR: {current_lr:.6f} | train loss: {avg_train_loss:.4f}, train acc: {avg_train_acc:.4f}, train time: {train_time:.1f}s, "

        # --- Validation Phase ---
        print(f"\n🔄 Starting validation...")
        avg_val_loss, avg_val_acc, val_auc, val_time = validate_one_epoch(
            model, val_loader, criterion, device, epoch, cfg['epoch'], args.use_amp
        )
        log_text += f"val loss: {avg_val_loss:.4f}, val acc: {avg_val_acc:.4f}, val auc: {val_auc:.4f}, val time: {val_time:.1f}s"
        logger.info(log_text)
        print(log_text)
     
        # --- W&B Logging ---
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/accuracy': avg_train_acc,
                'train/time_seconds': train_time,
                'val/loss': avg_val_loss,
                'val/accuracy': avg_val_acc,
                'val/auc': val_auc,
                'val/time_seconds': val_time,
                'learning_rate': current_lr
            })

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            print(f"✅ Validation loss improved to {best_val_loss:.4f}. Saving best model.")
            # Save the best model based on validation loss
            torch.save({
                'model': model.state_dict(),
                'val_auc': val_auc
                }, os.path.join(save_path, 'weights', 'best_model.tar'))
        else:
            early_stop_counter += 1
            print(f"⚠️ Validation loss did not improve ({best_val_loss:.4f}). Counter: {early_stop_counter}/{args.early_stop_patience}")
        
        if early_stop_counter >= args.early_stop_patience:
            print(f"🛑 Early stopping triggered after {args.early_stop_patience} epochs with no improvement.")
            break

        # --- Atomic Checkpoint Saving ---
        checkpoint_path = os.path.join(save_path, 'weights', 'latest_checkpoint.tar')
        temp_checkpoint_path = checkpoint_path + ".tmp"
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'save_path': save_path,
            'wandb_run_id': wandb.run.id if not args.no_wandb else None,
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, temp_checkpoint_path)
        os.rename(temp_checkpoint_path, checkpoint_path)

    if not args.no_wandb:
        wandb.finish()
        
if __name__ == '__main__':
    # Set start method for multiprocessing to prevent potential CUDA issues
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the configuration file')
    parser.add_argument('-n', '--session-name', type=str, required=True, help='A name for the training session')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader (will use minimum 2)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config file')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Epochs to wait for val loss improvement before stopping')
    
    # --- New Performance Arguments ---
    parser.add_argument('--use-amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) for faster training')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Number of steps to accumulate gradients before an optimizer step')
    
    # --- W&B Arguments ---
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sbi-deepfake-detection', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username or team)')

    args = parser.parse_args()
    main(args)