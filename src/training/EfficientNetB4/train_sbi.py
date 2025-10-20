import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from datetime import datetime
from tqdm import tqdm
from model import Detector
import time

import os
import sys
# Ensure src is in sys.path for Docker/project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.utils.sbi import SBI_Dataset
from src.utils.scheduler import LinearDecayLR
from src.utils.logs import log
from src.utils.funcs import load_json

import wandb
import multiprocessing

def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    cfg=load_json(args.config)
    
    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.early_stop_patience is not None:
        cfg['early_stop_patience'] = args.early_stop_patience

    # Start a new run or resume an existing one
    wandb_run_id = None
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        wandb_run_id = checkpoint.get('wandb_run_id') 

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=cfg,
            id=wandb_run_id, 
            resume="allow"  
        )
        wandb.config.update(args, allow_val_change=True) 

    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    
    # SAM optimizer does 2 forward passes, so we need smaller batch size to fit in memory
    # Use the same batch size for train and val for consistency
    train_batch_size = batch_size // 2  # Halve for SAM's double forward pass
    val_batch_size = batch_size // 2    # Keep consistent with training
    
    train_dataset=SBI_Dataset(phase='train',image_size=image_size)
    val_dataset=SBI_Dataset(phase='val',image_size=image_size)
   
    # Force at least 2 workers for better performance
    actual_num_workers = max(2, args.num_workers)
    use_pin_memory = actual_num_workers > 0
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Train batch size: {train_batch_size}")
    print(f"   Val batch size: {val_batch_size}")
    print(f"   Num workers: {actual_num_workers} (requested: {args.num_workers})")
    print(f"   Pin memory: {use_pin_memory}\n")
    
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=actual_num_workers,
                        pin_memory=use_pin_memory,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn,
                        persistent_workers=True,
                        prefetch_factor=2  # Prefetch 2 batches per worker
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=val_batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=actual_num_workers,
                        pin_memory=use_pin_memory,
                        worker_init_fn=val_dataset.worker_init_fn,
                        persistent_workers=True,
                        prefetch_factor=2  # Prefetch 2 batches per worker
                        )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}\n")
    
    model=Detector(lr=cfg['lr']) # Pass learning rate from config
    model=model.to('cuda')
    
    # Don't compile - it causes delays in validation
    # try:
    #     model = torch.compile(model)
    #     print("✅ Model compiled with torch.compile()")
    # except Exception as e:
    #     print(f"⚠️ torch.compile() not available or failed: {e}")
    
    # --- 3. Watch the model with W&B ---
    if not args.no_wandb:
        wandb.watch(model, log='all', log_freq=100) # Log gradients and model topology
    
    n_epoch=cfg['epoch']
    lr_scheduler=LinearDecayLR(model.optimizer, n_epoch, int(n_epoch/4*3))
    
    start_epoch = 0
    weight_dict = {}
    last_val_auc = 0
    save_path = ''

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            # We already loaded the checkpoint to get the wandb_run_id
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            model.optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            weight_dict = checkpoint['weight_dict']
            last_val_auc = checkpoint['last_val_auc']
            save_path = checkpoint['save_path']
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            return

    if not save_path:
        now = datetime.now()
        save_path = 'output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path+'weights/', exist_ok=True)
        os.makedirs(save_path+'logs/', exist_ok=True)
        
    logger = log(path=save_path+"logs/", file="losses.logs")
    criterion=nn.CrossEntropyLoss()
    n_weight=5
    
    for epoch in range(start_epoch, n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train()
        
        # Empty cache before each epoch to prevent fragmentation
        torch.cuda.empty_cache()
        
        epoch_start = time.time()
        for step,data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")):
            img=data['img'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()
            
            # Forward pass through SAM optimizer (handles 2 passes internally)
            output=model.training_step(img, target)
            loss=criterion(output,target)
            train_loss+=loss.item()
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
            
            # Periodically empty cache to prevent memory fragmentation
            if step % 100 == 0 and step > 0:
                torch.cuda.empty_cache()
        
        train_time = time.time() - epoch_start
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        
        avg_train_loss = train_loss/len(train_loader)
        avg_train_acc = train_acc/len(train_loader)
        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, train time: {:.1f}s, ".format(
            epoch+1, n_epoch, avg_train_loss, avg_train_acc, train_time)

        # Validation
        print(f"\n🔄 Starting validation...")
        model.eval()
        val_loss=0.
        val_acc=0.
        output_dict=[]
        target_dict=[]
        np.random.seed(seed)
        
        # Empty cache and sync before validation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        val_start = time.time()
        with torch.no_grad():
            for step,data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")):
                if step == 0:
                    first_batch_time = time.time() - val_start
                    print(f"   First validation batch loaded in {first_batch_time:.2f}s")
                
                img=data['img'].to(device, non_blocking=True).float()
                target=data['label'].to(device, non_blocking=True).long()
                
                output=model(img)
                loss=criterion(output,target)
                
                val_loss+=loss.item()
                acc=compute_accuray(F.log_softmax(output,dim=1),target)
                val_acc+=acc
                output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
                target_dict+=target.cpu().data.numpy().tolist()
        
        val_time = time.time() - val_start
        avg_val_loss = val_loss/len(val_loader)
        avg_val_acc = val_acc/len(val_loader)
        val_auc=roc_auc_score(target_dict,output_dict)
        log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}, val time: {:.1f}s".format(
            avg_val_loss, avg_val_acc, val_auc, val_time)
     
        if len(weight_dict)<n_weight:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            weight_dict[save_model_path]=val_auc
            torch.save({ "model":model.state_dict() }, save_model_path.replace('_val.tar','_model.tar'))
            last_val_auc=min([weight_dict[k] for k in weight_dict])
        elif val_auc>=last_val_auc:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            for k in weight_dict:
                if weight_dict[k]==last_val_auc:
                    del weight_dict[k]
                    os.remove(k.replace('_val.tar','_model.tar'))
                    weight_dict[save_model_path]=val_auc
                    break
            torch.save({ "model":model.state_dict() }, save_model_path.replace('_val.tar','_model.tar'))
            last_val_auc=min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        print(log_text)

        # --- 4. Log Metrics to W&B ---
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

        # --- 5. Save W&B Run ID in Checkpoint ---
        checkpoint_path = os.path.join(save_path, 'weights', 'latest_checkpoint.tar')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': model.optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'weight_dict': weight_dict,
            'last_val_auc': last_val_auc,
            'save_path': save_path,
            'wandb_run_id': wandb.run.id if not args.no_wandb else None
        }, checkpoint_path)

    # --- 6. Finish the W&B Run ---
    if not args.no_wandb:
        wandb.finish()
        
if __name__=='__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # The start method can only be set once.

    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n', dest='session_name')
    parser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for DataLoader (will use minimum 2)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config file if provided)')
    parser.add_argument('--early-stop-patience', type=int, default=None, help='Early stopping patience in epochs')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='sbi-deepfake-detection', help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (username or team)')

    args=parser.parse_args()
    main(args)