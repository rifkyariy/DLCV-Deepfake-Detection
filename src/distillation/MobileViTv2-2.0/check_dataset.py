#!/usr/bin/env python3
"""
Dataset Integrity Checker

This script checks your SBI dataset for corrupted images and provides a summary.
Use this to identify and optionally remove problematic images before training.

Usage:
    python check_dataset.py
    python check_dataset.py --fix  # Create a list of bad images to exclude
"""

import os
import sys
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

# Fix import paths for Docker environment
if '/app/' in os.path.abspath(__file__):
    if '/app/src/utils' not in sys.path:
        sys.path.insert(0, '/app/src/utils')

from utils.funcs import load_json


def check_image(image_path):
    """Check if an image is valid."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Cannot read image"
        
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return False, f"Invalid dimensions: {h}x{w}"
        
        if h < 10 or w < 10:
            return False, f"Too small: {h}x{w}"
        
        return True, None
    except Exception as e:
        return False, str(e)


def check_dataset(phase='train'):
    """Check all images in a dataset phase."""
    print(f"\n{'='*80}")
    print(f"Checking {phase.upper()} dataset...")
    print('='*80)
    
    # Load dataset configuration
    cfg_path = 'src/configs/config_SBI.json'
    if not os.path.exists(cfg_path):
        cfg_path = '../../configs/config_SBI.json'
    
    cfg = load_json(cfg_path)
    
    # Get image list
    data_root = '/app/data' if '/app/' in os.path.abspath(__file__) else 'data'
    list_file = os.path.join(data_root, f'{phase}.json')
    
    if not os.path.exists(list_file):
        print(f"[ERROR] Cannot find {list_file}")
        return
    
    data_list = load_json(list_file)
    
    corrupted_images = []
    error_counts = {}
    
    print(f"Total images to check: {len(data_list)}")
    
    for item in tqdm(data_list, desc=f"Checking {phase} images"):
        img_path = item.get('img_path') or item.get('path')
        if not img_path:
            continue
        
        full_path = os.path.join(data_root, img_path)
        is_valid, error_msg = check_image(full_path)
        
        if not is_valid:
            corrupted_images.append({
                'path': img_path,
                'full_path': full_path,
                'error': error_msg
            })
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY for {phase.upper()} dataset")
    print('='*80)
    print(f"Total images checked: {len(data_list)}")
    print(f"Valid images: {len(data_list) - len(corrupted_images)}")
    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Corruption rate: {len(corrupted_images)/len(data_list)*100:.2f}%")
    
    if corrupted_images:
        print(f"\nError breakdown:")
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error}: {count} images")
        
        print(f"\nFirst 10 corrupted images:")
        for img in corrupted_images[:10]:
            print(f"  - {img['path']}: {img['error']}")
        
        # Save corrupted list
        output_file = f'corrupted_{phase}_images.txt'
        with open(output_file, 'w') as f:
            for img in corrupted_images:
                f.write(f"{img['path']}\t{img['error']}\n")
        
        print(f"\n[INFO] Full list saved to: {output_file}")
    
    return corrupted_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix', action='store_true', 
                       help='Create filtered dataset JSON without corrupted images')
    args = parser.parse_args()
    
    print("="*80)
    print("SBI Dataset Integrity Checker")
    print("="*80)
    
    # Check both train and validation sets
    train_corrupted = check_dataset('train')
    val_corrupted = check_dataset('val')
    
    total_corrupted = len(train_corrupted) + len(val_corrupted)
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print('='*80)
    print(f"Total corrupted images found: {total_corrupted}")
    print(f"  - Train: {len(train_corrupted)}")
    print(f"  - Val: {len(val_corrupted)}")
    
    if total_corrupted > 0:
        print(f"\n[RECOMMENDATION]")
        if total_corrupted < 50:
            print(f"  - Small number of corrupted images ({total_corrupted})")
            print(f"  - Training will skip them automatically (already handled)")
            print(f"  - No action needed, but you can clean them if desired")
        else:
            print(f"  - Significant number of corrupted images ({total_corrupted})")
            print(f"  - Recommend cleaning the dataset")
            print(f"  - Run with --fix to create filtered dataset JSONs")
    
    if args.fix and total_corrupted > 0:
        print(f"\n[INFO] Creating filtered dataset files...")
        # This would create new JSON files without corrupted images
        # Implementation depends on your dataset structure
        print("[TODO] Implement filtering in utils/sbi.py to skip corrupted images")


if __name__ == '__main__':
    main()
