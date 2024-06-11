parent_dir="/home/vmoskov/swiftnet/datasets/cityscapes/gtFine/train"

# Find all files in subdirectories and move them to the parent directory
find "$parent_dir" -mindepth 2 -type f -exec mv -t "$parent_dir" {} +

# Optionally, remove empty subdirectories
find "$parent_dir" -type d -empty -delete