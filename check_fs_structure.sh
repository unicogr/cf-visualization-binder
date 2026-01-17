#!/bin/bash

FS_DIR="/media/ng281432/Crucial X6/UNICOG/LePetitePrince/cf-visualization-binder/data/output/fs_subjects/sub-01_ses-01_iso"

echo "Checking FreeSurfer structure for: $FS_DIR"
echo "================================================================"

# Check what exists
for dir in surf mri label scripts stats; do
    if [ -d "$FS_DIR/$dir" ]; then
        echo "✅ $dir/ exists"
        ls "$FS_DIR/$dir" | head -5
    else
        echo "❌ $dir/ MISSING"
    fi
    echo ""
done

# List all files
echo "All files in subject directory:"
find "$FS_DIR" -type f | sort
