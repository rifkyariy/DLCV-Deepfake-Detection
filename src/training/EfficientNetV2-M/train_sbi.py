import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import the new scheduler
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from datetime import datetime
from tqdm import tqdm
import wandb
import multiprocessing

# --- Import model module ---
from model import Detector

# import src/utils 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.sbi import SBI_Dataset
from utils.logs import log
from utils.funcs import load_json

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*The 'repr' attribute with value.*has no effect.*"
)

def compute_accuray(pred,true):
    """Computes the accuracy of predictions."""
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    
    cfg=load_json(args.config)

    # --- Weights & Biases Initialization ---
    wandb_run_id = None
    if args.resume and os.path.isfile(args.resume):
        try:
            checkpoint = torch.load(args.resume)
            wandb_run_id = checkpoint.get('wandb_run_id')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("The checkpoint file seems corrupted. Please delete it and restart.")
            return
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

    # --- Reproducibility and Device Setup ---
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    # --- Dataset and DataLoader Configuration ---
    image_size=cfg['image_size']
    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']
    
    train_dataset=SBI_Dataset(phase='train',image_size=image_size)
    val_dataset=SBI_Dataset(phase='val',image_size=image_size)
   
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    
    # --- Model, Optimizer, and Scheduler Setup ---
    model=Detector(lr=cfg['lr'])
    model=model.to('cuda')
    
    if not args.no_wandb:
        wandb.watch(model, log='all', log_freq=100)
    
    n_epoch=cfg['epoch']
    # Use ReduceLROnPlateau scheduler to adapt learning rate based on validation loss
    lr_scheduler = ReduceLROnPlateau(model.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # --- Checkpoint Resuming Logic ---
    start_epoch = 0
    weight_dict = {}
    save_path = ''
    n_weight = 5
    best_val_loss = float('inf')
    early_stop_counter = 0

    if args.resume and checkpoint:
        print(f"Loading checkpoint '{args.resume}'")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        save_path = checkpoint['save_path']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        print(f"Resumed training from epoch {start_epoch}")
        
        print("Re-synchronizing top models from files...")
        weights_dir = os.path.join(save_path, 'weights')
        for f in os.listdir(weights_dir):
            if f.endswith('_model.tar'):
                try:
                    parts = f.split('_')
                    auc = float(parts[1])
                    full_path = os.path.join(weights_dir, f)
                    weight_dict[full_path] = auc
                except (IndexError, ValueError):
                    continue
        
        print(f"Found {len(weight_dict)} existing top models. Cleaning up if necessary...")
        while len(weight_dict) > n_weight:
            worst_model_path = min(weight_dict, key=weight_dict.get)
            
            if os.path.isfile(worst_model_path):
                print(f"Removing deprecated model {worst_model_path} with AUC: {weight_dict[worst_model_path]:.4f}")
                os.remove(worst_model_path)
            
            del weight_dict[worst_model_path]
        print(f"Cleanup complete. {len(weight_dict)} top models remain.")

    elif args.resume and not checkpoint:
        return

    # --- Setup Output Directories for a New Run ---
    if not save_path:
        now = datetime.now()
        save_path = 'output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'weights/', exist_ok=True)
        os.makedirs(save_path+'logs/', exist_ok=True)
        
    logger = log(path=save_path+"logs/", file="losses.logs")
    criterion=nn.CrossEntropyLoss()
    
    # --- Main Training Loop ---
    for epoch in range(start_epoch, n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train(mode=True)
        
        # --- Training Phase ---
        for step,data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            output=model.training_step(img, target)
            loss=criterion(output,target)
            train_loss+=loss.item()
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        current_lr = model.optimizer.param_groups[0]['lr'] # Get current LR from optimizer
        
        avg_train_loss = train_loss/len(train_loader)
        avg_train_acc = train_acc/len(train_loader)
        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(epoch+1, n_epoch, avg_train_loss, avg_train_acc)

        # --- Validation Phase ---
        model.train(mode=False)
        val_loss=0.
        val_acc=0.
        output_dict=[]
        target_dict=[]
        np.random.seed(seed)
        for step,data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            
            with torch.no_grad():
                output=model(img)
                loss=criterion(output,target)
            
            val_loss+=loss.item()
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            val_acc+=acc
            output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
            target_dict+=target.cpu().data.numpy().tolist()

        avg_val_loss = val_loss/len(val_loader)
        avg_val_acc = val_acc/len(val_loader)
        val_auc=roc_auc_score(target_dict,output_dict)
        log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(avg_val_loss, avg_val_acc, val_auc)
     
        # --- Step the new LR scheduler with the validation loss ---
        lr_scheduler.step(avg_val_loss)
     
        # --- Top-5 Model Saving Logic ---
        is_top_5 = len(weight_dict) < n_weight
        if not is_top_5:
            worst_auc = min(weight_dict.values())
            if val_auc >= worst_auc:
                is_top_5 = True
        
        if is_top_5:
            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_model.tar")
            
            print(f"Saving new top model to {save_model_path} with AUC: {val_auc:.4f}")
            torch.save({"model": model.state_dict()}, save_model_path)
            
            weight_dict[save_model_path] = val_auc
            
            if len(weight_dict) > n_weight:
                worst_model_path = min(weight_dict, key=weight_dict.get)
                
                if os.path.isfile(worst_model_path):
                    print(f"Removing worst model {worst_model_path} with AUC: {weight_dict[worst_model_path]:.4f}")
                    os.remove(worst_model_path)
                
                del weight_dict[worst_model_path]
        
        logger.info(log_text)

        # --- W&B Logging ---
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/accuracy': avg_train_acc,
                'val/loss': avg_val_loss,
                'val/accuracy': avg_val_acc,
                'val/auc': val_auc,
                'learning_rate': current_lr
            })

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f}. Resetting counter.")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve ({best_val_loss:.4f}). Counter: {early_stop_counter}/{args.early_stop_patience}")
        
        if early_stop_counter >= args.early_stop_patience:
            print(f"Early stopping triggered after {args.early_stop_patience} epochs with no improvement.")
            break # Exit the training loop

        # --- Atomic Checkpoint Saving ---
        checkpoint_path = os.path.join(save_path, 'weights', 'latest_checkpoint.tar')
        temp_checkpoint_path = checkpoint_path + ".tmp"
        
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'weight_dict': weight_dict,
            'save_path': save_path,
            'wandb_run_id': wandb.run.id if not args.no_wandb else None,
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter
        }, temp_checkpoint_path)

        os.rename(temp_checkpoint_path, checkpoint_path)

    # --- Finalize W&B Run ---
    if not args.no_wandb:
        wandb.finish()
        
if __name__=='__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Argument Parsing ---
    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for DataLoader')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config file')
    
    # --- New argument for Early Stopping ---
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Number of epochs to wait for validation loss improvement before stopping')
    
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sbi-deepfake-detection', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username or team)')

    args=parser.parse_args()
    main(args)

