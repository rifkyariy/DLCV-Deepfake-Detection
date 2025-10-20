# save this as src/debug_loader.py

import sys
# Make sure the script can find your utility files
sys.path.append('/app/src/')

from utils.sbi import SBI_Dataset
from tqdm import tqdm

print("Initializing validation dataset to find all bad files...")

# Use the same parameters as in your training script
val_dataset = SBI_Dataset(phase='val', image_size=224)

print(f"Checking {len(val_dataset)} files in the validation set...")

bad_files = []

# Loop through every single item in the dataset
for i in tqdm(range(len(val_dataset))):
    try:
        # We access the filename before trying to load the data
        filename = val_dataset.image_list[i]
        
        # This line will trigger the loading and potential crash
        _ = val_dataset[i] 
        
    except Exception as e:
        # If an error occurs, log the file and continue to the next one
        print(f"\n--- 💥 Found a problematic file ---")
        print(f"Index: {i}")
        print(f"File: {filename}")
        print(f"Error: {e}")
        bad_files.append(filename)

print("\n--- ✅ Debug script finished ---")
if bad_files:
    print(f"Found {len(bad_files)} problematic files. You should delete them.")
    for f in bad_files:
        print(f)
else:
    print("No files raised exceptions. The issue might be a silent crash.")