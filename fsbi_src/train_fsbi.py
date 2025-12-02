import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import gc
import wandb
from glob import glob
from utils.esbi import ESBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector

def compute_accuray(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    tmp = pred_idx == true.cpu().numpy()
    return sum(tmp) / len(pred_idx)

def main(args):
    cfg = load_json(args.config)

    # --- 1. WANDB INIT ---
    # Picks up env vars from your bash script automatically
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "FSBI-Deepfake-Detection"),
        name=os.getenv("WANDB_NAME", f"{args.session_name}_{args.mode}"),
        config=cfg,
        resume="allow"
    )

    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_size = cfg['image_size']
    
    # --- 2. BATCH SIZE FIX ---
    # We use the FULL batch size defined in config to match the paper (16).
    # Reducing this to 8 breaks SAM optimizer generalization.
    batch_size = cfg['batch_size'] 
    
    print(f"\nConfiguration Check:")
    print(f"▶ Image Size: {image_size}x{image_size} (Matches Paper)")
    print(f"▶ Batch Size: {batch_size} (Matches Paper - Critical for SAM)")
    print(f"▶ Epochs:     {cfg['epoch']} (Ensure this is > 75 for generalization)\n")

    # Datasets
    train_dataset_esbi = ESBI_Dataset(phase='train', image_size=image_size, wavelet=args.wavelet, mode=args.mode)
    val_dataset_esbi = ESBI_Dataset(phase='val', image_size=image_size, wavelet=args.wavelet, mode=args.mode)
   
    train_loader_esbi = torch.utils.data.DataLoader(
        train_dataset_esbi,
        batch_size=batch_size, # FIXED: No longer dividing by 2
        shuffle=True,
        collate_fn=train_dataset_esbi.collate_fn,
        num_workers=4, # Reduced to 4 to save RAM for the larger batch
        pin_memory=True,
        drop_last=True,
        worker_init_fn=train_dataset_esbi.worker_init_fn,
        prefetch_factor=2
    )
    
    val_loader_esbi = torch.utils.data.DataLoader(
        val_dataset_esbi,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset_esbi.collate_fn,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=val_dataset_esbi.worker_init_fn,
        prefetch_factor=2
    )
    
    model = Detector()
    model = model.to(device)
    
    scaler = torch.cuda.amp.GradScaler()

    # Resume Logic
    resume_checkpoint = None
    start_epoch = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    weight_dict = {}
    last_val_auc = 0
    n_weight = 5

    if args.resume:
        checkpoint_pattern = 'output/{}_*'.format(args.session_name)
        matching_dirs = sorted(glob(checkpoint_pattern))
        if matching_dirs:
            latest_dir = matching_dirs[-1]
            checkpoint_files = glob(os.path.join(latest_dir, 'weights', '*.tar'))
            if checkpoint_files:
                regular_checkpoints = [f for f in checkpoint_files if not os.path.basename(f).startswith('emergency')]
                if regular_checkpoints:
                    resume_checkpoint = max(regular_checkpoints, key=lambda x: int(os.path.basename(x).split('_')[0]))

    now = datetime.now()
    if resume_checkpoint:
        save_path = os.path.dirname(os.path.dirname(resume_checkpoint)) + '/'
        logger = log(path=save_path+"logs/", file="losses.logs")
    else:
        save_path = 'output/{}_'.format(args.session_name) + now.strftime(os.path.splitext(os.path.basename(args.config))[0]) + '_' + now.strftime("%m_%d_%H_%M_%S") + '/'
        os.mkdir(save_path)
        os.mkdir(save_path+'weights/')
        os.mkdir(save_path+'logs/')
        logger = log(path=save_path+"logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()
    n_epoch = cfg['epoch']
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch, int(n_epoch*0.75))
    
    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
        
        existing_weights = glob(os.path.join(save_path, 'weights', '*.tar'))
        for weight_path in existing_weights:
            try:
                auc = float(os.path.basename(weight_path).split('_')[1].replace('.tar', ''))
                weight_dict[weight_path] = auc
            except:
                pass
        if weight_dict:
            last_val_auc = min([weight_dict[k] for k in weight_dict])
            
        for _ in range(start_epoch):
            lr_scheduler.step()
        
        print(f"Resuming from epoch {start_epoch}/{n_epoch}")

    t_loader = train_loader_esbi
    v_loader = val_loader_esbi
    
    print("🚀 Training started. Monitor at https://wandb.ai")

    for epoch in range(start_epoch, n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.
        train_acc = 0.
        model.train()
        
        # Training Loop
        batches_processed = 0
        for step, data in enumerate(tqdm(t_loader, desc=f"Epoch {epoch+1} Train")):
            try:
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()
                
                try:
                    output = model.training_step(img, target)
                    
                    loss_value = criterion(output, target).item()
                    train_loss += loss_value
                    acc = compute_accuray(F.log_softmax(output.float(), dim=1), target)
                    train_acc += acc
                    batches_processed += 1
                    
                    if step % 10 == 0:
                        wandb.log({"batch_loss": loss_value, "batch_acc": acc})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # OOM Handling: Clean and Skip
                        print(f"\n⚠️ OOM at step {step}. Skipping batch.")
                        torch.cuda.empty_cache()
                        gc.collect()
                        del img, target
                        continue 
                    else:
                        raise e 
            
            except Exception as e:
                print(f"Error in data loading: {e}")
                continue
        
        # Avoid division by zero if all batches fail (unlikely)
        if batches_processed > 0:
            epoch_train_loss = train_loss/batches_processed
            epoch_train_acc = train_acc/batches_processed
        else:
            epoch_train_loss = 0
            epoch_train_acc = 0
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        lr_scheduler.step()
        torch.cuda.empty_cache()

        # Validation Loop
        model.eval()
        val_acc, val_loss = 0., 0.
        output_dict, target_dict = [], []  
        np.random.seed(seed)
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(v_loader, desc=f"Epoch {epoch+1} Val")):
                img = data['img'].to(device, non_blocking=True).float()
                target = data['label'].to(device, non_blocking=True).long()
                
                output = model(img)
                loss = criterion(output, target)
                
                loss_value = loss.item()
                val_loss += loss_value
                acc = compute_accuray(F.log_softmax(output, dim=1), target)
                val_acc += acc
                
                output_dict.extend(output.softmax(1)[:,1].cpu().data.numpy().tolist())
                target_dict.extend(target.cpu().data.numpy().tolist())
                
        epoch_val_loss = val_loss/len(v_loader)
        epoch_val_acc = val_acc/len(v_loader)
        val_auc = roc_auc_score(target_dict, output_dict)

        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        # Log to W&B
        wandb.log({
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_auc": val_auc,
            "lr": lr_scheduler.get_lr()[0],
            "epoch": epoch + 1
        })

        log_text = "Epoch {}/{} | train loss: {:.4f}, val loss: {:.4f}, val auc: {:.4f}".format(
            epoch+1, n_epoch, epoch_train_loss, epoch_val_loss, val_auc)
        
        logger.info(log_text)
        print(log_text)

        # Save Logic
        save_model_path = os.path.join(save_path+'weights/', "{}_{:.4f}_val.tar".format(epoch+1, val_auc))
        save_payload = {
            "model": model.state_dict(),
            "optimizer": model.optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        if len(weight_dict) < n_weight:
            weight_dict[save_model_path] = val_auc
            torch.save(save_payload, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])

        elif val_auc >= last_val_auc:
            for k in weight_dict:
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path] = val_auc
                    break
            torch.save(save_payload, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])
            
    wandb.finish()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    parser.add_argument('-w', dest='wavelet')
    parser.add_argument('-m', dest='mode')
    parser.add_argument('-e', dest='epoch', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args)