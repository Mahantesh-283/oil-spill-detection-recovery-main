import os
import shutil

# 1. Define source paths (your current mixed folders)
palsar_source = r"E:/oil spill1/train/palsar"
sentinel_source = r"E:/oil spill1/train/sentinel"

# 2. Define base organized folder
base_dir = r"E:/oil spill1/organized_train"
# Create target directories
subfolders = [
    "palsar/sat", "palsar/mask",
    "sentinel/sat", "sentinel/mask"
]
for folder in subfolders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

def sort_files(source, sensor_name):
    print(f"Sorting {sensor_name} files...")
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        if os.path.isdir(file_path): continue
        
        # Determine if it's a mask or satellite image
        # Adjust keywords 'mask' or 'gt' based on your actual filenames
        if "mask" in filename.lower() or "gt" in filename.lower():
            target = os.path.join(base_dir, sensor_name, "mask")
        else:
            target = os.path.join(base_dir, sensor_name, "sat")
            
        shutil.move(file_path, os.path.join(target, filename))

# Run for both sensors
sort_files(palsar_source, "palsar")
sort_files(sentinel_source, "sentinel")

print("Data separation complete.")