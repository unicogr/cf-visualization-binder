#!/usr/bin/env python
"""Create minimal dummy MRI files for FreeSurfer structure."""

import nibabel as nib
import numpy as np
from pathlib import Path

BASE_DIR = Path("/media/ng281432/Crucial X6/UNICOG/LePetitePrince/cf-visualization-binder/data/output/fs_subjects")

print("Creating dummy MRI files...")

for subject_dir in BASE_DIR.glob("sub-*"):
    if subject_dir.is_dir():
        print(f"Processing: {subject_dir.name}")
        
        mri_dir = subject_dir / "mri"
        mri_dir.mkdir(exist_ok=True)
        
        # Create dummy orig.mgz (256x256x256 volume)
        orig_file = mri_dir / "orig.mgz"
        if not orig_file.exists():
            # Create a small dummy volume (to save space)
            dummy_data = np.zeros((256, 256, 256), dtype=np.uint8)
            affine = np.eye(4)
            img = nib.MGHImage(dummy_data, affine)
            nib.save(img, orig_file)
            print(f"  ✅ Created {orig_file.name}")
        
        # Create dummy brain.mgz
        brain_file = mri_dir / "brain.mgz"
        if not brain_file.exists():
            dummy_data = np.zeros((256, 256, 256), dtype=np.uint8)
            affine = np.eye(4)
            img = nib.MGHImage(dummy_data, affine)
            nib.save(img, brain_file)
            print(f"  ✅ Created {brain_file.name}")

print("✅ All dummy MRI files created!")
