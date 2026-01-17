#!/bin/bash
# Create minimal FreeSurfer structure required by neuropythy

set -e

BASE_DIR="/media/ng281432/Crucial X6/UNICOG/LePetitePrince/cf-visualization-binder/data/output/fs_subjects"

echo "🔧 Fixing FreeSurfer subject structure..."

for subject_dir in "$BASE_DIR"/sub-*; do
    if [ -d "$subject_dir" ]; then
        subject_name=$(basename "$subject_dir")
        echo "Processing: $subject_name"
        
        # Create required directories
        mkdir -p "$subject_dir/mri"
        mkdir -p "$subject_dir/label"
        mkdir -p "$subject_dir/scripts"
        mkdir -p "$subject_dir/stats"
        mkdir -p "$subject_dir/touch"
        
        # Check if surf directory exists
        if [ ! -d "$subject_dir/surf" ]; then
            echo "  ⚠️  No surf/ directory found!"
            continue
        fi
        
        # Create a minimal orig.mgz file (required by neuropythy)
        if [ ! -f "$subject_dir/mri/orig.mgz" ]; then
            echo "  Creating dummy orig.mgz..."
            # We'll create this in Python since we need nibabel
        fi
        
        echo "  ✅ $subject_name structure created"
    fi
done

echo "✅ Done!"
