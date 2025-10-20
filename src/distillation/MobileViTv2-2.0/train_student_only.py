#!/usr/bin/env python3
"""
Stage 1: Pre-train Student Model (MobileViTv2-2.0)

This script trains the MobileViTv2-2.0 student model on your dataset WITHOUT distillation.
This creates a stable initialization before applying knowledge distillation.

Usage:
    python train_student_only.py configs/config_SBI.json -n pretrain_mobilevit
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast

# Fix import paths for Docker environment
if '/app/' in os.path.abspath(__file__):
    if '/app/src/utils' not in sys.path:
        sys.path.insert(0, '/app/src/utils')
    if '/app/src' not in sys.path:
        sys.path.insert(0, '/app/src')

from student_model import Detector as StudentDetector
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from utils.logs import log
from utils.funcs import load_json


def compute_accuracy(pred, true):
    pred_idx = pred.argmax(dim=1)
    return (pred_idx == true).float().mean().item()


def main(args):
    cfg = load_json(args.config)

    # --- Seeding ---
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- DataLoaders ---
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    
    # Use larger batch size for student-only training (no teacher overhead)
    student_batch_size = batch_size // 2  # Half of original for SAM
    print(f"[INFO] Using batch size: {student_batch_size} (reduced from {batch_size} for SAM optimizer)")
    
    train_dataset = SBI_Dataset(phase='train', image_size=image_size)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=student_batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=student_batch_size, 
        shuffle=False, 
        collate_fn=val_dataset.collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"SBI(train): {len(train_dataset)}")
    print(f"SBI(val): {len(val_dataset)}")
    
    # --- Model Setup ---
    lr = cfg.get('lr', 0.0001)  # Use conservative LR for pre-training
    print(f"Creating student model (MobileViTv2-2.0) with lr={lr}...")
    student_model = StudentDetector(lr=lr)
    student_model = student_model.to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(student_model.net, 'set_grad_checkpointing'):
        student_model.net.set_grad_checkpointing(enable=True)
        print("[INFO] Enabled gradient checkpointing for student model")
    
    # --- Optimizer and Scheduler ---
    optimizer = student_model.optimizer
    n_epoch = cfg['epoch']
    lr_scheduler = LinearDecayLR(optimizer, n_epoch, int(n_epoch / 4 * 3))
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(init_scale=512.0, growth_interval=100)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # --- Logging and Saving Setup ---
    now = datetime.now()
    save_path = f'output/{args.session_name}_{now.strftime(os.path.splitext(os.path.basename(args.config))[0])}_{now.strftime("%m_%d_%H_%M_%S")}/'
    os.makedirs(save_path + 'weights/', exist_ok=True)
    os.makedirs(save_path + 'logs/', exist_ok=True)
    logger = log(path=save_path + "logs/", file="losses.logs")
    
    weight_dict = {}
    n_weight = 5
    last_val_auc = 0
    
    print("=" * 80)
    print("STAGE 1: Pre-training Student Model (MobileViTv2-2.0)")
    print("=" * 80)
    print(f"[INFO] Training configuration:")
    print(f"  - Model: MobileViTv2-2.0 (18M parameters)")
    print(f"  - Optimizer: SAM (Sharpness-Aware Minimization)")
    print(f"  - Learning Rate: {lr}")
    print(f"  - Batch Size: {student_batch_size}")
    print(f"  - Epochs: {n_epoch}")
    print(f"  - Loss: CrossEntropy (no distillation)")
    print("=" * 80)
    
    torch.cuda.empty_cache()
    
    for epoch in range(n_epoch):
        # --- TRAINING LOOP ---
        train_loss = 0.
        train_acc = 0.
        student_model.train()
        
        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            
            # SAM requires two forward-backward passes
            # First forward-backward pass
            with autocast():
                output = student_model(img)
                loss = criterion(output, target)
            
            # Check for NaN before backprop
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARN] NaN/Inf loss detected at step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backward pass for first step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Check gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n[WARN] NaN/Inf gradient detected at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                continue
            
            # First step: ascend to adversarial point
            optimizer.first_step(zero_grad=True)
            scaler.update()
            
            # Second forward-backward pass
            with autocast():
                output = student_model(img)
                loss2 = criterion(output, target)
            
            # Check second loss
            if torch.isnan(loss2) or torch.isinf(loss2):
                print(f"\n[WARN] NaN/Inf loss in second pass at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                continue
            
            # Backward pass for second step
            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping for second pass
            grad_norm2 = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm2) or torch.isinf(grad_norm2):
                print(f"\n[WARN] NaN/Inf gradient in second pass at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                continue
            
            # Second step: update weights
            optimizer.second_step(zero_grad=True)
            scaler.update()
            
            train_loss += loss2.item()
            train_acc += compute_accuracy(output, target)
            
            # Clear cache periodically
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        lr_scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        log_text = f"Epoch {epoch+1}/{n_epoch} | train loss: {avg_train_loss:.4f}, train acc: {avg_train_acc:.4f}, "
        
        # --- VALIDATION LOOP ---
        student_model.eval()
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")):
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()
                
                with autocast():
                    output = student_model(img)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                val_acc += compute_accuracy(output, target)
                
                # Get predictions
                preds = output.softmax(1)[:, 1].cpu().data.numpy()
                if np.isnan(preds).any() or np.isinf(preds).any():
                    print(f"[WARN] NaN/Inf in predictions at step {step}, replacing with 0.5")
                    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
                
                output_dict.extend(preds.tolist())
                target_dict.extend(target.cpu().data.numpy().tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Clean output_dict if needed
        if any(np.isnan(x) or np.isinf(x) for x in output_dict):
            print(f"[WARN] Cleaning NaN/Inf values in output_dict")
            output_dict = [0.5 if (np.isnan(x) or np.isinf(x)) else x for x in output_dict]
        
        val_auc = roc_auc_score(target_dict, output_dict)
        log_text += f"val loss: {avg_val_loss:.4f}, val acc: {avg_val_acc:.4f}, val auc: {val_auc:.4f}"
        
        # --- Model Saving Logic ---
        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_val.pth")
            weight_dict[save_model_path] = val_auc
            torch.save({
                "model": student_model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "epoch": epoch,
                "val_auc": val_auc
            }, save_model_path)
            last_val_auc = min(weight_dict.values())
            
        elif val_auc > last_val_auc:
            # Remove worst model
            worst_model_path = min(weight_dict, key=weight_dict.get)
            if os.path.exists(worst_model_path):
                os.remove(worst_model_path)
            del weight_dict[worst_model_path]
            
            # Add new best model
            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_val.pth")
            weight_dict[save_model_path] = val_auc
            torch.save({
                "model": student_model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "epoch": epoch,
                "val_auc": val_auc
            }, save_model_path)
            last_val_auc = min(weight_dict.values())
        
        logger.info(log_text)
        print(log_text)
    
    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETE: Pre-training Finished")
    print("=" * 80)
    print(f"Best models saved in: {save_path}weights/")
    print(f"Best validation AUC: {max(weight_dict.values()):.4f}")
    print("\nNext Steps:")
    print("1. Find the best checkpoint (highest AUC) in the weights folder")
    print("2. Update config_SBI.json with:")
    print(f'   "pretrained_student_weights": "{save_path}weights/BEST_MODEL.pth"')
    print("3. Run knowledge distillation:")
    print("   python main_distillation.py configs/config_SBI.json -n distill_mobilevit")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to the config file")
    parser.add_argument('-n', '--session_name', default='pretrain_mobilevit', 
                       help="Session name for output folder")
    args = parser.parse_args()
    main(args)
