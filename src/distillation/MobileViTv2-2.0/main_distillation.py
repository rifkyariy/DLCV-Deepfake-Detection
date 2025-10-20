import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
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

# --- Custom Imports from your project ---
from teacher_model import Detector as TeacherDetector
from student_model import Detector as StudentDetector
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from utils.logs import log
from utils.funcs import load_json

def get_student_model(lr, device, pretrained_student_path=None):
    """
    Creates the MobileViTv2-2.0 student model with Apple's pretrained ImageNet-21k-1k weights.
    
    Args:
        lr (float): Learning rate for the SAM optimizer
        device: The device to load the model on
        pretrained_student_path: Optional path to load a pre-finetuned student model
    """
    print("Creating student model (MobileViTv2-2.0)...")
    
    # Use the StudentDetector from student_model.py which handles:
    # - Loading MobileViTv2-2.0 from timm
    # - Apple pretrained weights
    # - Custom classifier for binary classification
    # - SAM optimizer initialization
    student_model = StudentDetector(lr=lr)
    student_model = student_model.to(device)
    
    # If a pre-finetuned student checkpoint is provided, load it
    if pretrained_student_path and os.path.exists(pretrained_student_path):
        print(f"[INFO] Loading pre-finetuned student weights from {pretrained_student_path}")
        checkpoint = torch.load(pretrained_student_path, map_location=device)
        
        if 'model' in checkpoint:
            student_model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            student_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            student_model.load_state_dict(checkpoint, strict=False)
        
        print("[INFO] ✓ Pre-finetuned student weights loaded successfully")
    else:
        print("[INFO] Starting from ImageNet pretrained weights (no task-specific finetuning)")
    
    return student_model

def get_teacher_model(weight_path, device):
    """Loads the pre-trained EfficientNet-B4 teacher model from weights/FFc23.tar."""
    print(f"Loading teacher model (EfficientNet-B4) from {weight_path}...")

    # Use the TeacherDetector from teacher_model.py
    teacher_model = TeacherDetector()
    teacher_model = teacher_model.to(device)

    checkpoint = torch.load(weight_path, map_location=device)

    # Extract state_dict from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Check if keys have 'net.' prefix
    sample_keys = list(state_dict.keys())[:5]
    has_net_prefix = any(key.startswith('net.') for key in sample_keys)

    if has_net_prefix:
        # Keys have 'net.' prefix, load into full model
        print("Loading checkpoint with 'net.' prefix into full Detector model")
        teacher_model.load_state_dict(state_dict, strict=False)
    else:
        # Keys don't have prefix, load directly into teacher_model.net
        print("Loading checkpoint into teacher_model.net (EfficientNet)")
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not k.startswith('optimizer') and not k.startswith('epoch')}
        teacher_model.net.load_state_dict(filtered_state_dict, strict=False)

    print(f"✓ Teacher model loaded successfully from {weight_path}")
    teacher_model.eval()  # Set teacher to evaluation mode

    # Freeze teacher parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model

def compute_accuracy(pred, true):
    # For efficiency, keep tensors on GPU for comparison
    pred_idx = pred.argmax(dim=1)
    return (pred_idx == true).float().mean().item()

def main(args):
    cfg = load_json(args.config)

    # --- Seeding and Device Setup ---
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- DataLoaders ---
    image_size = cfg['image_size']
    batch_size = cfg['batch_size']
    
    # Reduce batch size for distillation (teacher + student + SAM = high memory)
    # SAM does 2 forward passes, so effective batch is already doubled
    distill_batch_size = batch_size // 4  # Reduce to 1/4 for distillation
    print(f"[INFO] Using batch size: {distill_batch_size} (reduced from {batch_size} for distillation)")
    
    train_dataset = SBI_Dataset(phase='train', image_size=image_size)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=distill_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=distill_batch_size, shuffle=False, collate_fn=val_dataset.collate_fn, num_workers=4, pin_memory=True)
    
    # --- Model Setup ---
    teacher_model = get_teacher_model(cfg['teacher_weights'], device)
    
    # Check if we should load a pre-finetuned student model for better initialization
    pretrained_student = cfg.get('pretrained_student_weights', None)
    student_model = get_student_model(lr=cfg['lr'], device=device, pretrained_student_path=pretrained_student)
    
    # Enable gradient checkpointing for student to save memory
    if hasattr(student_model.net, 'set_grad_checkpointing'):
        student_model.net.set_grad_checkpointing(enable=True)
        print("[INFO] Enabled gradient checkpointing for student model")
    
    # --- Optimizer, Scheduler, and Loss Functions ---
    # The student model has SAM optimizer initialized with the config lr
    optimizer = student_model.optimizer
    n_epoch = cfg['epoch']
    lr_scheduler = LinearDecayLR(optimizer, n_epoch, int(n_epoch / 4 * 3))
    
    # REFINED: Initialize GradScaler for AMP with conservative settings
    scaler = GradScaler(init_scale=512.0, growth_interval=100)

    # Distillation hyperparameters with safer defaults
    temp = cfg.get('temperature', 3.0)  # Lower temperature for stability
    alpha_start = cfg.get('alpha_start', 0.95)  # Start with mostly hard labels
    alpha_end = cfg.get('alpha', 0.5)  # End with balanced distillation
    
    # Progressive distillation: gradually increase soft label influence
    use_progressive = cfg.get('use_progressive_distillation', True)
    progressive_epochs = cfg.get('progressive_epochs', 20)  # Number of epochs to transition

    # REFINED: Simplified KL Divergence Loss
    distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
    student_loss_fn = nn.CrossEntropyLoss()

    # --- Logging and Saving Setup ---
    now = datetime.now()
    save_path = f'output/{args.session_name}_{now.strftime(os.path.splitext(os.path.basename(args.config))[0])}_{now.strftime("%m_%d_%H_%M_%S")}/'
    os.makedirs(save_path + 'weights/', exist_ok=True)
    os.makedirs(save_path + 'logs/', exist_ok=True)
    logger = log(path=save_path + "logs/", file="losses.logs")
    
    weight_dict = {}
    n_weight = 5
    last_val_auc = 0

    print("Starting Knowledge Distillation Training (Teacher: EffNet-b4, Student: MobileViTv2-2.0) with AMP...")
    print(f"[INFO] Memory optimization enabled: batch_size={distill_batch_size}, gradient_checkpointing=True")
    print(f"[INFO] Distillation config: lr={cfg['lr']}, temp={temp}, alpha_start={alpha_start}, alpha_end={alpha_end}")
    print(f"[INFO] Progressive distillation: {use_progressive}, transition over {progressive_epochs} epochs")
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Track consecutive NaN batches for early stopping
    consecutive_nan_batches = 0
    max_consecutive_nans = 50  # Stop if we get 50 consecutive NaN batches
    
    for epoch in range(n_epoch):
        # --- TRAINING LOOP (Distillation) ---
        train_loss = 0.
        train_acc = 0.
        student_model.train()
        
        # Progressive distillation: gradually decrease alpha (increase soft label influence)
        if use_progressive and epoch < progressive_epochs:
            current_alpha = alpha_start - (alpha_start - alpha_end) * (epoch / progressive_epochs)
        else:
            current_alpha = alpha_end
        
        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train] α={current_alpha:.2f}")):
            img = data['img'].to(device, non_blocking=True).float()
            target = data['label'].to(device, non_blocking=True).long()
            
            # SAM requires two forward-backward passes
            # First forward-backward pass - compute gradients
            with autocast():
                # Get teacher's predictions (no gradients needed)
                with torch.no_grad():
                    teacher_logits = teacher_model(img)

                # Get student's predictions
                student_logits = student_model(img)
                
                # Calculate distillation loss
                soft_teacher_probs = F.softmax(teacher_logits / temp, dim=1)
                soft_student_log_probs = F.log_softmax(student_logits / temp, dim=1)
                distillation_loss = distillation_loss_fn(soft_student_log_probs, soft_teacher_probs) * (temp ** 2)

                # Calculate student's own loss with ground truth
                student_cross_entropy_loss = student_loss_fn(student_logits, target)
                
                # Combine the losses with warmup alpha
                loss = current_alpha * student_cross_entropy_loss + (1.0 - current_alpha) * distillation_loss
            
            # Backward pass for first step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping to prevent explosion (more conservative)
            grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Check for NaN gradients before optimizer step
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"\n[WARN] NaN/Inf gradient detected at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nans:
                    print(f"\n[ERROR] Too many consecutive NaN batches ({consecutive_nan_batches}). Stopping training.")
                    print("[SOLUTION] The student model needs pre-training on your dataset first.")
                    print("Please run finetuning (without distillation) before attempting knowledge distillation.")
                    raise RuntimeError("Training failed - too many NaN batches")
                continue
            
            # First step: ascend to find adversarial weights (w + e(w))
            optimizer.first_step(zero_grad=True)
            
            # Update scaler after first step (required before second forward pass)
            scaler.update()
            
            # Second forward-backward pass at the adversarial point
            with autocast():
                # Get teacher's predictions again
                with torch.no_grad():
                    teacher_logits = teacher_model(img)
                
                # Get student's predictions at adversarial point
                student_logits = student_model(img)
                
                # Check student logits for NaN before computing loss
                if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                    print(f"\n[WARN] NaN/Inf in student logits at step {step}, skipping batch")
                    optimizer.zero_grad()
                    scaler.update()  # Must update scaler before continuing
                    consecutive_nan_batches += 1
                    if consecutive_nan_batches >= max_consecutive_nans:
                        print(f"\n[ERROR] Too many consecutive NaN batches ({consecutive_nan_batches}). Stopping training.")
                        print("[SOLUTION] The student model needs pre-training on your dataset first.")
                        print("Please run finetuning (without distillation) before attempting knowledge distillation.")
                        raise RuntimeError("Training failed - too many NaN batches")
                    continue
                
                # Calculate distillation loss
                soft_teacher_probs = F.softmax(teacher_logits / temp, dim=1)
                soft_student_log_probs = F.log_softmax(student_logits / temp, dim=1)
                distillation_loss = distillation_loss_fn(soft_student_log_probs, soft_teacher_probs) * (temp ** 2)

                # Calculate student's own loss with ground truth
                student_cross_entropy_loss = student_loss_fn(student_logits, target)
                
                # Combine the losses with progressive alpha
                loss2 = current_alpha * student_cross_entropy_loss + (1.0 - current_alpha) * distillation_loss
            
            # Check loss before backward
            if torch.isnan(loss2) or torch.isinf(loss2):
                print(f"\n[WARN] NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nans:
                    print(f"\n[ERROR] Too many consecutive NaN batches ({consecutive_nan_batches}). Stopping training.")
                    print("[SOLUTION] The student model needs pre-training on your dataset first.")
                    print("Please run finetuning (without distillation) before attempting knowledge distillation.")
                    raise RuntimeError("Training failed - too many NaN batches")
                continue
            
            # Backward pass for second step
            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping for second pass as well
            grad_norm2 = torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Check gradients again
            if torch.isnan(grad_norm2) or torch.isinf(grad_norm2):
                print(f"\n[WARN] NaN/Inf gradient in second pass at step {step}, skipping batch")
                optimizer.zero_grad()
                scaler.update()  # Must update scaler before continuing
                consecutive_nan_batches += 1
                if consecutive_nan_batches >= max_consecutive_nans:
                    print(f"\n[ERROR] Too many consecutive NaN batches ({consecutive_nan_batches}). Stopping training.")
                    print("[SOLUTION] The student model needs pre-training on your dataset first.")
                    print("Please run finetuning (without distillation) before attempting knowledge distillation.")
                    raise RuntimeError("Training failed - too many NaN batches")
                continue
            
            # Second step: descend from adversarial point back to original and update
            optimizer.second_step(zero_grad=True)
            scaler.update()
            
            # Reset consecutive NaN counter on successful batch
            consecutive_nan_batches = 0
            
            train_loss += loss2.item()
            train_acc += compute_accuracy(student_logits, target)
            
            # Clear cache more frequently for distillation
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        lr_scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        log_text = f"Epoch {epoch+1}/{n_epoch} | train loss: {avg_train_loss:.4f}, train acc: {avg_train_acc:.4f}, "

        # --- VALIDATION LOOP (on Student Model) ---
        student_model.eval()
        val_loss = 0.
        val_acc = 0.
        output_dict = []
        target_dict = []
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")):
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()
                
                # Validation can also use autocast for a slight speedup
                with autocast():
                    output = student_model(img)
                    loss = student_loss_fn(output, target)
                
                val_loss += loss.item()
                val_acc += compute_accuracy(output, target)
                
                # Get predictions and check for NaN
                preds = output.softmax(1)[:, 1].cpu().data.numpy()
                if np.isnan(preds).any() or np.isinf(preds).any():
                    print(f"[WARN] NaN/Inf detected in predictions at step {step}, replacing with 0.5")
                    preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
                
                output_dict.extend(preds.tolist())
                target_dict.extend(target.cpu().data.numpy().tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Check for NaN in output_dict before computing AUC
        if any(np.isnan(x) or np.isinf(x) for x in output_dict):
            print(f"[ERROR] NaN/Inf still present in output_dict after cleaning")
            output_dict = [0.5 if (np.isnan(x) or np.isinf(x)) else x for x in output_dict]
        
        val_auc = roc_auc_score(target_dict, output_dict)
        log_text += f"val loss: {avg_val_loss:.4f}, val acc: {avg_val_acc:.4f}, val auc: {val_auc:.4f}"
    
        # --- Model Saving Logic (for Student Model) ---
        if len(weight_dict) < n_weight:
            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_val.pth")
            weight_dict[save_model_path] = val_auc
            torch.save({"model": student_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, save_model_path)
            last_val_auc = min(weight_dict.values())

        elif val_auc > last_val_auc:
            # Find and remove the worst model
            worst_model_path = min(weight_dict, key=weight_dict.get)
            if os.path.exists(worst_model_path):
                os.remove(worst_model_path)
            del weight_dict[worst_model_path]
            
            # Add the new best model
            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_val.pth")
            weight_dict[save_model_path] = val_auc
            torch.save({"model": student_model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, save_model_path)
            last_val_auc = min(weight_dict.values())
        
        logger.info(log_text)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to the config file")
    parser.add_argument('-n', '--session_name', default='distillation_mobilevitv2_20', help="Session name for output folder") 
    args = parser.parse_args()
    main(args)