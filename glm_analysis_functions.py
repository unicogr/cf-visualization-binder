"""
GLM Group-Level Random Effects Analysis Functions

This module provides functions for multi-subject GLM analysis including:
- Data aggregation across subjects
- ROI-eccentricity binning
- KL divergence computation
- Monte Carlo permutation testing
- Hierarchical Bayesian modeling
- Mixed effects modeling
- Group visualization

Author: Nicolas Gravel (nicolas.gravel@cea.fr)
Date: February 12, 2026
Project: Congenital Blindness study (3T)
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import warnings
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pickle
import gc

# Optional imports with graceful fallback
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    # Define placeholder for type hints
    az = None

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import neuropythy as ny
except ImportError:
    ny = None


def aggregate_subjects_roi_data(
    subject_ids: List[str],
    task: str,
    contrast_name: str,
    roi_names: List[str],
    data_root: Path,
    benson_dir: Path,
    surf_dir: Path,
    ecc_bins: np.ndarray = None,
    hemisphere: str = 'both',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Aggregate GLM contrast data across multiple subjects with ROI and eccentricity information.
    
    Parameters
    ----------
    subject_ids : List[str]
        List of subject IDs (e.g., ['03', '05', '09'])
    task : str
        Task name (e.g., 'lpp')
    contrast_name : str
        Name of GLM contrast (e.g., 'math-vs-language')
    roi_names : List[str]
        List of ROI names to include (e.g., ['V1', 'V2', 'V3'])
    data_root : Path
        Root directory containing GLM outputs
    benson_dir : Path
        Directory containing Benson atlas data
    surf_dir : Path
        Directory containing surface files
    ecc_bins : np.ndarray, optional
        Eccentricity bin edges (default: np.arange(0.5, 12.5, 0.5))
    hemisphere : str, optional
        'lh', 'rh', or 'both' (default: 'both')
    verbose : bool, optional
        Print progress messages (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject_id, roi, hemisphere, vertex_idx, 
                                eccentricity, ecc_bin, t_value
    """
    if ecc_bins is None:
        ecc_bins = np.arange(0.5, 12.5, 0.5)
    
    hemis = ['lh', 'rh'] if hemisphere == 'both' else [hemisphere]
    
    all_data = []
    
    for subject_id in subject_ids:
        if verbose:
            print(f"Processing subject {subject_id}...")
        
        for hemi in hemis:
            try:
                # Load GLM contrast map from combined contrasts pickle file
                # Note: derivatives is sibling to output, so use .parent to go up one level
                glm_dir = data_root.parent / 'derivatives' / 'glm-first-level'
                combined_file = glm_dir / f"sub-{subject_id}_ses-01_task-mathlang_combined_contrasts.pkl"
                
                if not combined_file.exists():
                    if verbose:
                        print(f"  ⚠️  GLM combined contrasts file not found: {combined_file}")
                    continue
                
                # Load combined contrasts pickle
                with open(combined_file, 'rb') as f:
                    combined_contrasts = pickle.load(f)
                
                # Check if requested contrast exists
                if contrast_name not in combined_contrasts:
                    if verbose:
                        print(f"  ⚠️  Contrast '{contrast_name}' not found in combined contrasts")
                        print(f"      Available: {list(combined_contrasts.keys())}")
                    continue
                
                # Extract stat (t-statistic) for this contrast
                stat_img = combined_contrasts[contrast_name]['stat']
                
                # Extract hemisphere-specific data from SurfaceImage
                try:
                    if hasattr(stat_img, 'data') and isinstance(stat_img.data, dict):
                        lh_data = stat_img.data['left']
                        rh_data = stat_img.data['right']
                    elif hasattr(stat_img, 'data') and hasattr(stat_img.data, 'parts'):
                        lh_data = stat_img.data.parts['left']
                        rh_data = stat_img.data.parts['right']
                    else:
                        # Try direct mesh access (PolyData)
                        lh_data = stat_img.mesh.parts['left'].point_data['stat']
                        rh_data = stat_img.mesh.parts['right'].point_data['stat']
                    
                    t_values = lh_data if hemi == 'lh' else rh_data
                    
                    # Convert t-statistics to z-scores for valid group-level inference
                    # For large df (typical in fMRI), t ≈ z, but we standardize for safety
                    z_values = (t_values - np.nanmean(t_values)) / np.nanstd(t_values)
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Error extracting data for contrast '{contrast_name}': {e}")
                    continue
                
                # Load Benson atlas (.curv files)
                surf_dir = benson_dir / f'sub-{subject_id}_ses-01_iso' / 'surf'
                atlas_path = surf_dir / f'{hemi}.benson14_varea.curv'
                if not atlas_path.exists():
                    if verbose:
                        print(f"  ⚠️  Benson atlas not found: {atlas_path}")
                    continue
                
                roi_labels = nib.freesurfer.io.read_morph_data(str(atlas_path))
                
                # Load eccentricity map (.curv file)
                ecc_path = surf_dir / f'{hemi}.benson14_eccen.curv'
                if not ecc_path.exists():
                    if verbose:
                        print(f"  ⚠️  Eccentricity map not found: {ecc_path}")
                    continue
                
                eccentricity = nib.freesurfer.io.read_morph_data(str(ecc_path))
                
                # ROI name to label mapping (Benson atlas convention)
                roi_label_map = {'V1': 1, 'V2': 2, 'V3': 3, 'hV4': 4, 'VO1': 5, 'VO2': 6, 
                                'LO1': 7, 'LO2': 8, 'TO1': 9, 'TO2': 10, 'V3b': 11, 'V3a': 12}
                
                # Process each requested ROI
                for roi_name in roi_names:
                    if roi_name not in roi_label_map:
                        if verbose:
                            print(f"  ⚠️  Unknown ROI: {roi_name}")
                        continue
                    
                    roi_label = roi_label_map[roi_name]
                    roi_mask = roi_labels == roi_label
                    
                    if not np.any(roi_mask):
                        continue
                    
                    # Get data for this ROI
                    roi_vertices = np.where(roi_mask)[0]
                    roi_z_values = z_values[roi_mask]
                    roi_eccentricity = eccentricity[roi_mask]
                    
                    # Filter invalid values
                    valid_mask = (np.isfinite(roi_z_values) & 
                                 np.isfinite(roi_eccentricity) & 
                                 (roi_eccentricity > 0) & 
                                 (roi_eccentricity < 15))
                    
                    if not np.any(valid_mask):
                        continue
                    
                    # Bin eccentricity
                    ecc_bin_indices = np.digitize(roi_eccentricity[valid_mask], ecc_bins)
                    ecc_bin_centers = (ecc_bins[:-1] + ecc_bins[1:]) / 2
                    ecc_bin_values = np.zeros(len(roi_eccentricity[valid_mask]))
                    for i, bin_idx in enumerate(ecc_bin_indices):
                        if 0 < bin_idx < len(ecc_bin_centers):
                            ecc_bin_values[i] = ecc_bin_centers[bin_idx - 1]
                    
                    # Create records
                    for vertex, z_val, ecc, ecc_bin in zip(
                        roi_vertices[valid_mask],
                        roi_z_values[valid_mask],
                        roi_eccentricity[valid_mask],
                        ecc_bin_values
                    ):
                        all_data.append({
                            'subject_id': subject_id,
                            'roi': roi_name,
                            'hemisphere': hemi,
                            'vertex_idx': vertex,
                            'eccentricity': ecc,
                            'ecc_bin': ecc_bin,
                            't_value': z_val  # Now contains z-score
                        })
                
                if verbose:
                    print(f"  ✅ {hemi}: Loaded {len([d for d in all_data if d['subject_id'] == subject_id and d['hemisphere'] == hemi])} vertices")
                    
            except Exception as e:
                if verbose:
                    print(f"  ❌ Error processing {subject_id}/{hemi}: {e}")
                continue
    
    if not all_data:
        raise ValueError("No valid data found for any subject")
    
    df = pd.DataFrame(all_data)
    
    if verbose:
        print(f"\n✅ Total: {len(df)} vertices across {len(subject_ids)} subjects")
        print(f"   ROIs: {df['roi'].unique().tolist()}")
        print(f"   Subjects: {df['subject_id'].unique().tolist()}")
    
    return df


def bin_vertices_by_roi_ecc(
    group_df: pd.DataFrame,
    ecc_bin_width: float = 0.5
) -> pd.DataFrame:
    """
    Bin vertices by ROI and eccentricity, computing summary statistics.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    ecc_bin_width : float, optional
        Width of eccentricity bins (default: 0.5 degrees)
        
    Returns
    -------
    pd.DataFrame
        Binned data with columns: subject_id, roi, ecc_bin, n_vertices, 
                                  mean_t, std_t, median_t
    """
    binned_data = []
    
    for (subject_id, roi), grp in group_df.groupby(['subject_id', 'roi']):
        for ecc_bin in grp['ecc_bin'].unique():
            bin_data = grp[grp['ecc_bin'] == ecc_bin]
            
            if len(bin_data) > 0:
                binned_data.append({
                    'subject_id': subject_id,
                    'roi': roi,
                    'ecc_bin': ecc_bin,
                    'n_vertices': len(bin_data),
                    'mean_t': bin_data['t_value'].mean(),
                    'std_t': bin_data['t_value'].std(),
                    'median_t': bin_data['t_value'].median()
                })
    
    return pd.DataFrame(binned_data)


def compute_kl_divergence_per_roi(
    group_df: pd.DataFrame,
    roi1: str = None,
    roi2: str = None,
    normalize: bool = True,
    epsilon: float = 1e-10
) -> Dict[str, float]:
    """
    Compute KL divergence between eccentricity distributions for ROI pairs.
    
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi1 : str, optional
        First ROI (if None, computes for all pairs)
    roi2 : str, optional
        Second ROI (if None, computes for all pairs)
    normalize : bool, optional
        Normalize distributions to sum to 1 (default: True)
    epsilon : float, optional
        Small value to avoid log(0) (default: 1e-10)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with ROI pairs as keys and KL divergence as values
    """
    kl_results = {}
    
    rois = group_df['roi'].unique()
    
    if roi1 and roi2:
        roi_pairs = [(roi1, roi2)]
    else:
        roi_pairs = [(r1, r2) for i, r1 in enumerate(rois) for r2 in rois[i+1:]]
    
    for r1, r2 in roi_pairs:
        df1 = group_df[group_df['roi'] == r1]
        df2 = group_df[group_df['roi'] == r2]
        
        if len(df1) == 0 or len(df2) == 0:
            continue
        
        # Get eccentricity distributions
        ecc_bins = np.sort(group_df['ecc_bin'].unique())
        
        # Count vertices in each bin
        counts1 = np.array([len(df1[df1['ecc_bin'] == b]) for b in ecc_bins])
        counts2 = np.array([len(df2[df2['ecc_bin'] == b]) for b in ecc_bins])
        
        if normalize:
            p = counts1 / (counts1.sum() + epsilon)
            q = counts2 / (counts2.sum() + epsilon)
        else:
            p = counts1
            q = counts2
        
        # Add epsilon to avoid division by zero
        p = p + epsilon
        q = q + epsilon
        
        # Compute KL divergence
        kl_div = np.sum(p * np.log(p / q))
        
        kl_results[f'{r1}-{r2}'] = kl_div
    
    return kl_results


def kl_divergence_monte_carlo_null(
    group_df: pd.DataFrame,
    roi_pair: Tuple[str, str],
    n_permutations: int = 10000,
    verbose: bool = True
) -> float:
    """
    Compute p-value for KL divergence using Monte Carlo permutation testing.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_pair : Tuple[str, str]
        Tuple of two ROI names (e.g., ('V1', 'V2'))
    n_permutations : int, optional
        Number of permutations (default: 10000)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    float
        P-value (proportion of null KL divergences >= observed)
    """
    roi1, roi2 = roi_pair
    
    # Compute observed KL divergence
    observed_kl = compute_kl_divergence_per_roi(group_df, roi1, roi2)
    observed_value = observed_kl[f'{roi1}-{roi2}']
    
    if verbose:
        print(f"Observed KL({roi1}||{roi2}) = {observed_value:.4f}")
        print(f"Running {n_permutations} permutations...")
    
    # Get data for both ROIs
    df_combined = group_df[group_df['roi'].isin([roi1, roi2])].copy()
    
    null_kls = []
    
    for i in range(n_permutations):
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_permutations}")
        
        # Shuffle ROI labels WITHIN subjects to preserve subject structure
        shuffled_df = df_combined.copy()
        for subject_id in df_combined['subject_id'].unique():
            subj_mask = shuffled_df['subject_id'] == subject_id
            shuffled_df.loc[subj_mask, 'roi'] = np.random.permutation(
                shuffled_df.loc[subj_mask, 'roi'].values
            )
        
        # Compute KL for permuted data
        null_kl = compute_kl_divergence_per_roi(shuffled_df, roi1, roi2)
        if f'{roi1}-{roi2}' in null_kl:
            null_kls.append(null_kl[f'{roi1}-{roi2}'])
    
    null_kls = np.array(null_kls)
    
    # Compute p-value
    p_value = np.mean(null_kls >= observed_value)
    
    if verbose:
        print(f"✅ P-value = {p_value:.4f}")
        print(f"   Null mean = {null_kls.mean():.4f}, std = {null_kls.std():.4f}")
    
    # Memory cleanup: delete large arrays created during permutation testing
    try:
        del null_kls
        gc.collect()
    except NameError:
        pass
    
    return p_value


def within_roi_kl_divergence_test(
    group_df: pd.DataFrame,
    roi_name: str,
    n_permutations: int = 100,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Test if within-ROI eccentricity distribution is different from shuffled null via KL divergence.
    
    Computes KL divergence between observed eccentricity bins and shuffled null distribution.
    Shuffles eccentricity labels within subjects to test if the observed distribution
    structure is greater than chance.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_name : str
        ROI name to test
    n_permutations : int, optional
        Number of permutations (default: 100)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with observed KL divergence, p-value, and null distribution stats
    """
    from scipy.stats import entropy
    
    df_roi = group_df[group_df['roi'] == roi_name].copy()
    
    if len(df_roi) < 10:
        if verbose:
            print(f"⚠️  Insufficient data for {roi_name}")
        return {'observed_kl': np.nan, 'p_value': np.nan}
    
    # Get eccentricity bins
    ecc_bins = np.sort(df_roi['ecc_bin'].unique())
    
    # Compute observed distribution
    obs_counts = np.array([len(df_roi[df_roi['ecc_bin'] == b]) for b in ecc_bins])
    obs_dist = obs_counts / obs_counts.sum()
    
    # Compute a uniform reference distribution
    uniform_dist = np.ones(len(ecc_bins)) / len(ecc_bins)
    
    # Observed KL divergence from uniform
    observed_kl = entropy(obs_dist, uniform_dist)
    
    if verbose:
        print(f"   {roi_name}: observed KL = {observed_kl:.4f}", end=' - ', flush=True)
    
    null_kls = []
    
    for i in range(n_permutations):
        # Shuffle eccentricity bins WITHIN subjects to preserve structure
        shuffled_df = df_roi.copy()
        for subject_id in df_roi['subject_id'].unique():
            subj_mask = shuffled_df['subject_id'] == subject_id
            shuffled_df.loc[subj_mask, 'ecc_bin'] = np.random.permutation(
                shuffled_df.loc[subj_mask, 'ecc_bin'].values
            )
        
        # Compute KL for permuted data
        null_counts = np.array([len(shuffled_df[shuffled_df['ecc_bin'] == b]) for b in ecc_bins])
        null_dist = null_counts / null_counts.sum()
        null_kl = entropy(null_dist, uniform_dist)
        null_kls.append(null_kl)
    
    null_kls = np.array(null_kls)
    
    # One-tailed p-value (is observed KL larger than null?)
    p_value = np.mean(null_kls >= observed_kl)
    
    if verbose:
        sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"p = {p_value:.4f} {sig_marker}")
    
    result = {
        'observed_kl': observed_kl,
        'p_value': p_value,
        'null_mean': null_kls.mean(),
        'null_std': null_kls.std()
    }
    
    # Memory cleanup: delete large arrays created during permutation testing
    try:
        del null_kls
        gc.collect()
    except NameError:
        pass
    
    return result


def within_roi_kl_vs_shuffled_ecc(
    group_df: pd.DataFrame,
    roi_name: str,
    n_permutations: int = 100,
    verbose: bool = True,
    progress_callback = None,
    ecc_range: tuple = None,
    hemisphere: str = None,
    inference_method: str = 'permutation'
) -> Dict[str, float]:
    """
    Test if within-ROI t-stat vs eccentricity relationship is significant via KL divergence.
    
    Computes KL divergence between the joint distribution of (t-stat, eccentricity) bins
    and a null distribution where eccentricity labels are shuffled within subjects.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_name : str
        ROI name to test
    n_permutations : int, optional
        Number of permutations (default: 100)
    verbose : bool, optional
        Print progress (default: True)
    progress_callback : callable, optional
        Callback function(current, total) for progress updates
    ecc_range : tuple, optional
        (min_ecc, max_ecc) to filter data before testing (default: None, use all data)
    hemisphere : str, optional
        'lh', 'rh', or None for both hemispheres combined (default: None)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with observed KL divergence, p-value, null distribution stats, and per-bin directional tests
    """
    from scipy.stats import entropy
    
    df_roi = group_df[group_df['roi'] == roi_name].copy()
    
    # Filter by hemisphere if specified
    if hemisphere is not None:
        df_roi = df_roi[df_roi['hemisphere'] == hemisphere].copy()
    
    # Filter by eccentricity range if specified
    if ecc_range is not None:
        min_ecc, max_ecc = ecc_range
        df_roi = df_roi[(df_roi['eccentricity'] >= min_ecc) & (df_roi['eccentricity'] <= max_ecc)]
    
    if len(df_roi) < 10:
        if verbose:
            hemi_str = f" ({hemisphere})" if hemisphere else ""
            print(f"⚠️  Insufficient data for {roi_name}{hemi_str}")
        return {
            'observed_kl': np.nan, 
            'p_value': np.nan, 
            'null_mean': np.nan, 
            'null_std': np.nan,
            'bin_results': None
        }
    
    # Create joint bins for (t-stat, eccentricity)
    # Bin t-stats into quartiles
    t_bin_edges = np.percentile(df_roi['t_value'], [0, 25, 50, 75, 100])
    t_bin_edges[-1] += 0.001  # Ensure max value is included
    df_roi['t_bin'] = np.digitize(df_roi['t_value'], t_bin_edges[:-1])
    
    # Use existing eccentricity bins
    ecc_bins = np.sort(df_roi['ecc_bin'].unique())
    
    # Compute empirical mean t-stat per eccentricity bin
    empirical_bin_means = []
    bin_centers = []
    
    for ecc_bin in ecc_bins:
        bin_mask = df_roi['ecc_bin'] == ecc_bin
        bin_data = df_roi[bin_mask]['t_value']
        
        if len(bin_data) > 0:
            empirical_bin_means.append(bin_data.mean())
            # Estimate bin center (assuming 1-degree bins)
            bin_centers.append(ecc_bin + 0.5)
        else:
            empirical_bin_means.append(np.nan)
            bin_centers.append(ecc_bin + 0.5)
    
    empirical_bin_means = np.array(empirical_bin_means)
    bin_centers = np.array(bin_centers)
    
    # Create joint distribution (t_bin x ecc_bin)
    def compute_joint_kl(df):
        """Compute KL divergence of joint (t-stat, ecc) distribution vs uniform."""
        joint_counts = np.zeros((len(np.unique(df['t_bin'])), len(ecc_bins)))
        
        for i, t_bin in enumerate(sorted(df['t_bin'].unique())):
            for j, ecc_bin in enumerate(ecc_bins):
                mask = (df['t_bin'] == t_bin) & (df['ecc_bin'] == ecc_bin)
                joint_counts[i, j] = np.sum(mask)
        
        # Flatten to 1D and normalize
        joint_dist = joint_counts.flatten()
        joint_dist = joint_dist / (joint_dist.sum() + 1e-10)
        
        # Uniform reference
        uniform_dist = np.ones_like(joint_dist) / len(joint_dist)
        
        # Add epsilon to avoid log(0)
        joint_dist = joint_dist + 1e-10
        uniform_dist = uniform_dist + 1e-10
        
        return entropy(joint_dist, uniform_dist)
    
    def compute_bin_means(df, ecc_bins_list):
        """Compute mean t-stat per eccentricity bin."""
        bin_means = []
        for ecc_bin in ecc_bins_list:
            bin_mask = df['ecc_bin'] == ecc_bin
            bin_data = df[bin_mask]['t_value']
            if len(bin_data) > 0:
                bin_means.append(bin_data.mean())
            else:
                bin_means.append(np.nan)
        return np.array(bin_means)
    
    # Observed KL divergence
    observed_kl = compute_joint_kl(df_roi)
    
    if verbose:
        print(f"   {roi_name}: observed KL = {observed_kl:.4f}", end=' - ', flush=True)
    
    # Permutation test - collect both KL and per-bin means
    null_kls = []
    null_bin_means_array = []  # shape: (n_permutations, n_bins)
    
    for i in range(n_permutations):
        if progress_callback:
            progress_callback(i + 1, n_permutations)
        
        # Shuffle eccentricity bins WITHIN subjects to preserve structure
        shuffled_df = df_roi.copy()
        for subject_id in df_roi['subject_id'].unique():
            subj_mask = shuffled_df['subject_id'] == subject_id
            shuffled_df.loc[subj_mask, 'ecc_bin'] = np.random.permutation(
                shuffled_df.loc[subj_mask, 'ecc_bin'].values
            )
        
        # Compute KL for permuted data
        null_kl = compute_joint_kl(shuffled_df)
        null_kls.append(null_kl)
        
        # Compute per-bin means for null distribution
        null_bin_means = compute_bin_means(shuffled_df, ecc_bins)
        null_bin_means_array.append(null_bin_means)
    
    null_kls = np.array(null_kls)
    null_bin_means_array = np.array(null_bin_means_array)  # shape: (n_permutations, n_bins)
    
    # One-tailed p-value for KL (is observed KL larger than null?)
    # Default permutation inference
    kl_p_value = np.mean(null_kls >= observed_kl)

    # Optionally perform signed-rank inference across subjects on KL divergences
    # This tests whether subjects consistently show KL larger than their within-subject null.
    if inference_method == 'signed_rank':
        from scipy.stats import wilcoxon

        subj_ids = df_roi['subject_id'].unique()
        subj_deltas = []
        subj_null_means = []
        subj_obs_kls = []

        # For each subject compute observed KL and subject-specific null mean
        for sid in subj_ids:
            df_sub = df_roi[df_roi['subject_id'] == sid]
            if len(df_sub) < 3:
                continue
            # compute observed KL for the subject
            try:
                obs_kl_sub = compute_joint_kl(df_sub)
            except Exception:
                continue

            # build subject-specific null by shuffling ecc_bin within that subject
            null_kls_sub = []
            for _ in range(n_permutations):
                shuffled_sub = df_sub.copy()
                shuffled_sub['ecc_bin'] = np.random.permutation(shuffled_sub['ecc_bin'].values)
                try:
                    null_k = compute_joint_kl(shuffled_sub)
                    null_kls_sub.append(null_k)
                except Exception:
                    continue

            if len(null_kls_sub) == 0:
                continue

            subj_null_mean = np.mean(null_kls_sub)
            subj_deltas.append(obs_kl_sub - subj_null_mean)
            subj_null_means.append(subj_null_mean)
            subj_obs_kls.append(obs_kl_sub)

        # If insufficient subject data, return nan p-value
        if len(subj_deltas) < 2:
            kl_p_value = np.nan
        else:
            try:
                stat, p_sr = wilcoxon(subj_deltas, alternative='greater')
                kl_p_value = p_sr
            except TypeError:
                # Older scipy may not support 'alternative' keyword; fall back to two-sided then halve
                stat, p_two = wilcoxon(subj_deltas)
                kl_p_value = p_two / 2.0

        # Overwrite null summary with subject-level null summary
        null_mean = np.mean(subj_null_means) if subj_null_means else np.nan
        null_std = np.std(subj_null_means) if subj_null_means else np.nan
        # attach subject-level diagnostics to return payload
        extra_subject_info = {
            'subject_ids': subj_ids.tolist() if hasattr(subj_ids, 'tolist') else list(subj_ids),
            'subject_obs_kls': subj_obs_kls,
            'subject_null_means': subj_null_means,
            'subject_deltas': subj_deltas
        }
    else:
        null_mean = null_kls.mean() if len(null_kls) > 0 else np.nan
        null_std = null_kls.std() if len(null_kls) > 0 else np.nan
        extra_subject_info = None
    
    # Per-bin statistics and p-values
    null_means_per_bin = np.nanmean(null_bin_means_array, axis=0)
    null_stds_per_bin = np.nanstd(null_bin_means_array, axis=0)
    
    bin_p_values = []
    bin_directions = []
    
    for bin_idx in range(len(ecc_bins)):
        emp_mean = empirical_bin_means[bin_idx]
        null_dist_bin = null_bin_means_array[:, bin_idx]
        
        # Remove NaNs from null distribution
        null_dist_bin = null_dist_bin[~np.isnan(null_dist_bin)]
        
        if np.isnan(emp_mean) or len(null_dist_bin) == 0:
            bin_p_values.append(np.nan)
            bin_directions.append('ns')
            continue
        
        # Two-tailed p-value using percentile method
        percentile = np.sum(null_dist_bin <= emp_mean) / len(null_dist_bin)
        p_val = 2 * min(percentile, 1 - percentile)
        bin_p_values.append(p_val)
        
        # Determine direction
        if p_val < 0.05:
            if emp_mean > np.mean(null_dist_bin):
                if p_val < 0.001:
                    bin_directions.append('High***')
                elif p_val < 0.01:
                    bin_directions.append('High**')
                else:
                    bin_directions.append('High*')
            else:
                if p_val < 0.001:
                    bin_directions.append('Low***')
                elif p_val < 0.01:
                    bin_directions.append('Low**')
                else:
                    bin_directions.append('Low*')
        else:
            bin_directions.append('ns')
    
    if verbose:
        sig_marker = '***' if (not np.isnan(kl_p_value) and kl_p_value < 0.001) else '**' if (not np.isnan(kl_p_value) and kl_p_value < 0.01) else '*' if (not np.isnan(kl_p_value) and kl_p_value < 0.05) else 'ns'
        try:
            print(f"p = {kl_p_value:.4f} {sig_marker}")
        except Exception:
            print(f"p = {kl_p_value} {sig_marker}")

    result = {
        'observed_kl': observed_kl,
        'p_value': kl_p_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'inference_method': inference_method,
        'bin_results': {
            'bin_centers': bin_centers.tolist(),
            'empirical_means': empirical_bin_means.tolist(),
            'null_means': null_means_per_bin.tolist(),
            'null_stds': null_stds_per_bin.tolist(),
            'p_values': bin_p_values,
            'directions': bin_directions
        }
    }

    if extra_subject_info is not None:
        result['subject_level'] = extra_subject_info

    # Memory cleanup: delete large arrays created during permutation testing
    # These are no longer needed after results are computed and stored in result dict
    try:
        del null_kls, null_bin_means_array
        if 'null_kls' in locals():
            del null_kls
        if 'null_bin_means_array' in locals():
            del null_bin_means_array
        if 'subj_deltas' in locals():
            del subj_deltas
        if 'subj_null_means' in locals():
            del subj_null_means
        if 'subj_obs_kls' in locals():
            del subj_obs_kls
        gc.collect()
    except NameError:
        # Variables may not exist in all code paths, skip cleanup
        pass

    return result


def eccentricity_permutation_test(
    group_df: pd.DataFrame,
    roi_name: str,
    n_permutations: int = 1000,
    verbose: bool = True,
    ecc_range: tuple = None
) -> Dict[str, float]:
    """
    Test if eccentricity-ROI relationship is significant via permutation.
    
    Shuffles eccentricity labels within each ROI to test if the observed
    correlation between eccentricity and t-values is greater than chance.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_name : str
        ROI name to test
    n_permutations : int, optional
        Number of permutations (default: 1000)
    verbose : bool, optional
        Print progress (default: True)
    ecc_range : tuple, optional
        (min_ecc, max_ecc) to filter data before testing (default: None, use all data)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with observed correlation and p-value
    """
    from scipy.stats import pearsonr
    
    df_roi = group_df[group_df['roi'] == roi_name].copy()
    
    # Filter by eccentricity range if specified
    if ecc_range is not None:
        min_ecc, max_ecc = ecc_range
        df_roi = df_roi[(df_roi['eccentricity'] >= min_ecc) & (df_roi['eccentricity'] <= max_ecc)]
    
    if len(df_roi) < 10:
        if verbose:
            print(f"⚠️  Insufficient data for {roi_name}")
        return {'observed_corr': np.nan, 'p_value': np.nan}
    
    # Compute observed correlation
    observed_corr, _ = pearsonr(df_roi['eccentricity'], df_roi['t_value'])
    
    if verbose:
        print(f"Testing {roi_name}: observed r = {observed_corr:.4f}")
        print(f"Running {n_permutations} permutations...")
    
    null_corrs = []
    
    for i in range(n_permutations):
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_permutations}")
        
        # Shuffle eccentricity labels WITHIN subjects to preserve structure
        shuffled_df = df_roi.copy()
        for subject_id in df_roi['subject_id'].unique():
            subj_mask = shuffled_df['subject_id'] == subject_id
            shuffled_df.loc[subj_mask, 'eccentricity'] = np.random.permutation(
                shuffled_df.loc[subj_mask, 'eccentricity'].values
            )
        
        # Compute correlation for permuted data
        null_corr, _ = pearsonr(shuffled_df['eccentricity'], shuffled_df['t_value'])
        null_corrs.append(null_corr)
    
    null_corrs = np.array(null_corrs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
    
    if verbose:
        print(f"✅ P-value = {p_value:.4f}")
        print(f"   Null mean = {null_corrs.mean():.4f}, std = {null_corrs.std():.4f}")
    
    result = {
        'observed_corr': observed_corr,
        'p_value': p_value,
        'null_mean': null_corrs.mean(),
        'null_std': null_corrs.std()
    }
    
    # Memory cleanup: delete large arrays created during permutation testing
    try:
        del null_corrs
        gc.collect()
    except NameError:
        pass
    
    return result


def fit_hierarchical_kl_model(
    group_df: pd.DataFrame,
    roi_names: List[str],
    n_samples: int = 2000,
    n_tune: int = 1000,
    verbose: bool = True
) -> Optional[object]:
    """
    Fit hierarchical Bayesian model for group-level analysis with eccentricity covariate.
    
    Model:
    ------
    Level 1 (Observations): z_ijk ~ Normal(μ_ijk, σ_within)
                            μ_ijk = (β_0j + u_i0j) + (β_1j + u_i1j) * ecc_k
    
    Level 2 (Subjects):     u_i0j ~ Normal(0, τ_0j)  [random intercepts]
                            u_i1j ~ Normal(0, τ_1j)  [random slopes]
    
    Level 3 (Population):   β_0j ~ Normal(0, 10)     [pop intercept per ROI]
                            β_1j ~ Normal(0, 5)      [pop slope per ROI]
                            τ_0j, τ_1j ~ HalfNormal(2.5)
                            σ_within ~ HalfNormal(5)
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_names : List[str]
        List of ROIs to include in model
    n_samples : int, optional
        Number of MCMC samples (default: 2000)
    n_tune : int, optional
        Number of tuning samples (default: 1000)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    Optional[az.InferenceData]
        ArviZ InferenceData object, or None if PyMC not available
    """
    if not PYMC_AVAILABLE:
        if verbose:
            print("⚠️  PyMC not available, skipping hierarchical model")
        return None
    
    # Filter to requested ROIs
    df_model = group_df[group_df['roi'].isin(roi_names)].copy()
    
    # Create subject and ROI indices
    subject_ids = df_model['subject_id'].unique()
    subject_idx = pd.Categorical(df_model['subject_id'], categories=subject_ids).codes
    roi_idx = pd.Categorical(df_model['roi'], categories=roi_names).codes
    
    n_subjects = len(subject_ids)
    n_rois = len(roi_names)
    n_obs = len(df_model)
    
    # Extract eccentricity values
    ecc_values = df_model['eccentricity'].values
    
    if verbose:
        print(f"Fitting hierarchical Bayesian model with eccentricity:")
        print(f"  Subjects: {n_subjects}")
        print(f"  ROIs: {n_rois}")
        print(f"  Observations: {n_obs}")
    
    with pm.Model() as model:
        # Level 3: Population-level parameters
        beta_0 = pm.Normal('beta_0', mu=0, sigma=10, shape=n_rois)  # Intercepts
        beta_1 = pm.Normal('beta_1', mu=0, sigma=5, shape=n_rois)   # Slopes
        
        # Between-subject variability
        tau_0 = pm.HalfNormal('tau_0', sigma=2.5, shape=n_rois)  # Random intercept SD
        tau_1 = pm.HalfNormal('tau_1', sigma=2.5, shape=n_rois)  # Random slope SD
        
        # Within-subject variability
        sigma_within = pm.HalfNormal('sigma_within', sigma=5)
        
        # Level 2: Subject-specific deviations
        u_0 = pm.Normal('u_0', mu=0, sigma=1, shape=(n_subjects, n_rois))  # Non-centered
        u_1 = pm.Normal('u_1', mu=0, sigma=1, shape=(n_subjects, n_rois))  # Non-centered
        
        # Subject-specific intercepts and slopes
        alpha_subj = beta_0 + u_0 * tau_0  # Centered parameterization
        beta_subj = beta_1 + u_1 * tau_1
        
        # Level 1: Linear predictor
        mu = alpha_subj[subject_idx, roi_idx] + beta_subj[subject_idx, roi_idx] * ecc_values
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_within, observed=df_model['t_value'].values)
        
        # Derived quantities: ROI comparisons
        if n_rois >= 2:
            delta_slopes = pm.Deterministic('delta_slopes', 
                                           beta_1[0] - beta_1[1])  # V1 vs V2 slope difference
        
        # Sample
        if verbose:
            print(f"  Sampling {n_samples} iterations ({n_tune} tuning)...")
        
        trace = pm.sample(n_samples, tune=n_tune, 
                         target_accept=0.95,
                         return_inferencedata=True,
                         progressbar=verbose)
    
    if verbose:
        print("✅ Sampling complete")
        print(az.summary(trace, var_names=['beta_0', 'beta_1', 'tau_0', 'tau_1']))
    
    return trace


def fit_mixed_effects_model(
    group_df: pd.DataFrame,
    roi_names: List[str],
    verbose: bool = True
) -> Optional[Dict]:
    """
    Fit linear mixed effects model (frequentist approach) with random slopes.
    
    Model: z_ij = (β_0 + u_i0) + (β_1 + u_i1)*ecc + ε_ij
    
    Fixed effects: β_0 (intercept), β_1 (eccentricity slope)
    Random effects: u_i0 ~ N(0, σ_u0²) (subject-specific intercepts)
                    u_i1 ~ N(0, σ_u1²) (subject-specific slopes)
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_names : List[str]
        List of ROIs to include
    verbose : bool, optional
        Print results (default: True)
        
    Returns
    -------
    Optional[Dict]
        Dictionary with model results for each ROI, or None if statsmodels unavailable
    """
    if not STATSMODELS_AVAILABLE:
        if verbose:
            print("⚠️  Statsmodels not available, skipping mixed effects model")
        return None
    
    results = {}
    
    # Single-ROI models with random slopes
    for roi in roi_names:
        df_roi = group_df[group_df['roi'] == roi].copy()
        
        if len(df_roi) < 10:
            if verbose:
                print(f"⚠️  Insufficient data for {roi}")
            continue
        
        try:
            # Fit mixed effects model with random slopes
            model = mixedlm("t_value ~ eccentricity", df_roi, 
                          groups=df_roi["subject_id"],
                          re_formula="~eccentricity")  # Random slopes
            fitted = model.fit(reml=True)
            
            results[roi] = {
                'intercept': fitted.params['Intercept'],
                'slope': fitted.params['eccentricity'],
                'intercept_se': fitted.bse['Intercept'],
                'slope_se': fitted.bse['eccentricity'],
                'intercept_pval': fitted.pvalues['Intercept'],
                'slope_pval': fitted.pvalues['eccentricity'],
                'random_intercept_std': np.sqrt(fitted.cov_re.iloc[0, 0]),
                'random_slope_std': np.sqrt(fitted.cov_re.iloc[1, 1]) if fitted.cov_re.shape[0] > 1 else 0,
                'residual_std': np.sqrt(fitted.scale),
                'aic': fitted.aic,
                'bic': fitted.bic,
                'llf': fitted.llf
            }
            
            if verbose:
                print(f"\n{roi}:")
                print(f"  Intercept: {results[roi]['intercept']:.3f} ± {results[roi]['intercept_se']:.3f} (p={results[roi]['intercept_pval']:.4f})")
                print(f"  Slope: {results[roi]['slope']:.3f} ± {results[roi]['slope_se']:.3f} (p={results[roi]['slope_pval']:.4f})")
                print(f"  Random intercept σ: {results[roi]['random_intercept_std']:.3f}")
                print(f"  Random slope σ: {results[roi]['random_slope_std']:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"❌ Error fitting {roi}: {e}")
            continue
    
    # Multi-ROI model with interactions and hemisphere
    try:
        if verbose:
            print(f"\n📊 Fitting full model with ROI × Eccentricity × Hemisphere interactions...")
        
        # Prepare data for full model
        df_full = group_df[group_df['roi'].isin(roi_names)].copy()
        
        # Create categorical variables
        df_full['roi_cat'] = pd.Categorical(df_full['roi'])
        df_full['hemi_cat'] = pd.Categorical(df_full['hemisphere'])
        
        # Fit full model with interactions
        full_model = mixedlm(
            "t_value ~ eccentricity * C(roi_cat) * C(hemi_cat)",
            df_full,
            groups=df_full["subject_id"],
            re_formula="~eccentricity"
        )
        full_fitted = full_model.fit(reml=True, maxiter=200)
        
        results['full_model'] = {
            'params': full_fitted.params.to_dict(),
            'pvalues': full_fitted.pvalues.to_dict(),
            'aic': full_fitted.aic,
            'bic': full_fitted.bic,
            'llf': full_fitted.llf
        }
        
        if verbose:
            print(f"✅ Full model fit complete (AIC={full_fitted.aic:.1f}, BIC={full_fitted.bic:.1f})")
            
    except Exception as e:
        if verbose:
            print(f"⚠️  Full model failed: {e}")
    
    return results


def plot_group_roi_scatter(
    group_df: pd.DataFrame,
    subject_ids: List[str],
    roi_names: List[str],
    kl_divergences: Dict[str, Tuple[float, float]] = None,
    bin_results: Dict[str, Dict] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 22),
    alpha: float = 0.3,
    verbose: bool = True
) -> plt.Figure:
    """
    Create multi-panel shaded area plot showing group GLM results.
    Both hemispheres combined in each panel (LH=black, RH=red).
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    subject_ids : List[str]
        List of subject IDs to display
    roi_names : List[str]
        List of ROI names to plot
    kl_divergences : Dict[str, Tuple[float, float]], optional
        Dictionary mapping ROI pairs to (KL value, p-value)
    bin_results : Dict[str, Dict], optional
        Per-bin directional test results from within_roi_kl_vs_shuffled_ecc()
    output_path : Optional[Path], optional
        Path to save figure (default: None, don't save)
    figsize : Tuple[int, int], optional
        Figure size in inches (default: (18, 22))
    alpha : float, optional
        Shaded area transparency (default: 0.3)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_rois = len(roi_names)
    
    # Calculate global y-axis limits for consistent scaling
    y_min = group_df['t_value'].min()
    y_max = group_df['t_value'].max()
    y_range = y_max - y_min
    y_lim = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]
    
    # Create figure with 4 rows x 3 columns (both hemispheres combined)
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    
    # Process each ROI (both hemispheres on same plot)
    for idx, roi in enumerate(roi_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # LEFT HEMISPHERE (gray dots, black line)
        df_roi_lh = group_df[(group_df['roi'] == roi) & (group_df['hemisphere'] == 'lh')]
        
        if len(df_roi_lh) > 0:
            # Plot individual subject data points (gray)
            for subject_id in subject_ids:
                df_subj = df_roi_lh[df_roi_lh['subject_id'] == subject_id]
                if len(df_subj) > 0:
                    ax.scatter(df_subj['eccentricity'], df_subj['t_value'],
                              alpha=0.3, s=8, color='gray', zorder=3)
            
            # Compute group mean and SD for LH
            ecc_bins_lh = np.sort(df_roi_lh['ecc_bin'].unique())
            group_means_lh = []
            group_sds_lh = []
            
            for ecc_bin in ecc_bins_lh:
                bin_data = df_roi_lh[df_roi_lh['ecc_bin'] == ecc_bin]
                if len(bin_data) > 0:
                    group_means_lh.append(bin_data['t_value'].mean())
                    group_sds_lh.append(bin_data['t_value'].std())
                else:
                    group_means_lh.append(np.nan)
                    group_sds_lh.append(np.nan)
            
            # Convert to arrays
            group_means_lh = np.array(group_means_lh)
            group_sds_lh = np.array(group_sds_lh)
            
            # Calculate bin centers from bin edges (ecc_bins are left edges)
            # Assuming uniform bin width, center is edge + half of typical bin width
            if len(ecc_bins_lh) > 1:
                bin_width = np.diff(ecc_bins_lh).mean()
                ecc_centers_lh = ecc_bins_lh + bin_width / 2
            else:
                ecc_centers_lh = ecc_bins_lh
            
            # Plot LH mean curve (black)
            ax.plot(ecc_centers_lh, group_means_lh, color='black', linewidth=2.5, 
                   label='LH', zorder=10)
            
            # Show SD bands only for first plot (when no bin_results provided)
            if bin_results is None:
                ax.fill_between(ecc_centers_lh, 
                               group_means_lh - group_sds_lh,
                               group_means_lh + group_sds_lh,
                               color='black', alpha=alpha, zorder=5)
        
        # RIGHT HEMISPHERE (light red dots, red line)
        df_roi_rh = group_df[(group_df['roi'] == roi) & (group_df['hemisphere'] == 'rh')]
        
        if len(df_roi_rh) > 0:
            # Plot individual subject data points (light red)
            for subject_id in subject_ids:
                df_subj = df_roi_rh[df_roi_rh['subject_id'] == subject_id]
                if len(df_subj) > 0:
                    ax.scatter(df_subj['eccentricity'], df_subj['t_value'],
                              alpha=0.3, s=8, color='lightcoral', zorder=3)
            
            # Compute group mean and SD for RH
            ecc_bins_rh = np.sort(df_roi_rh['ecc_bin'].unique())
            group_means_rh = []
            group_sds_rh = []
            
            for ecc_bin in ecc_bins_rh:
                bin_data = df_roi_rh[df_roi_rh['ecc_bin'] == ecc_bin]
                if len(bin_data) > 0:
                    group_means_rh.append(bin_data['t_value'].mean())
                    group_sds_rh.append(bin_data['t_value'].std())
                else:
                    group_means_rh.append(np.nan)
                    group_sds_rh.append(np.nan)
            
            # Convert to arrays
            group_means_rh = np.array(group_means_rh)
            group_sds_rh = np.array(group_sds_rh)
            
            # Calculate bin centers from bin edges
            if len(ecc_bins_rh) > 1:
                bin_width = np.diff(ecc_bins_rh).mean()
                ecc_centers_rh = ecc_bins_rh + bin_width / 2
            else:
                ecc_centers_rh = ecc_bins_rh
            
            # Plot RH mean curve (red)
            ax.plot(ecc_centers_rh, group_means_rh, color='red', linewidth=2.5, 
                   label='RH', zorder=10)
            
            # Show SD bands only for first plot (when no bin_results provided)
            if bin_results is None:
                ax.fill_between(ecc_centers_rh, 
                               group_means_rh - group_sds_rh,
                               group_means_rh + group_sds_rh,
                               color='red', alpha=alpha, zorder=5)
        
        # Set x-axis limits to 11 degrees
        x_lim = [0, 11]
        
        # Styling with increased font sizes
        ax.set_xlabel('Eccentricity (deg)', fontsize=14)
        ax.set_ylabel('Z-score', fontsize=14)
        ax.set_title(f'{roi}', fontsize=16, fontweight='bold', color='black')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        # Legend removed for cleaner appearance
        
        # Set x-axis limits to 11 degrees
        x_lim = [0, 11]
        
        # Calculate dotted line positions within plot area (before setting y_lim)
        if bin_results:
            # Position High significance above 0, Low significance below 0
            # Use 25% of the positive range for High, 25% of negative range for Low
            pos_range = y_lim[1] - 0  # From 0 to max
            neg_range = 0 - y_lim[0]  # From 0 to min
            
            # High significance (positive excursions): above 0
            lh_high_y_pos = 0 + pos_range * 0.75  # 75% up from 0
            rh_high_y_pos = 0 + pos_range * 0.65   # 50% up from 0
            
            # Low significance (negative excursions): below 0  
            lh_low_y_pos = 0 - neg_range * 0.65    # 50% down from 0
            rh_low_y_pos = 0 - neg_range * 0.75   # 75% down from 0
            
            # No need to extend y_lim since positions are within existing bounds
            extended_y_lim = y_lim
        else:
            extended_y_lim = y_lim
        
        # Styling with increased font sizes
        ax.set_xlabel('Eccentricity (deg)', fontsize=14)
        ax.set_ylabel('Z-score', fontsize=14)
        ax.set_title(f'{roi}', fontsize=16, fontweight='bold', color='black')
        ax.set_xlim(x_lim)
        ax.set_ylim(extended_y_lim)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        # Legend removed for cleaner appearance
        
        # Overlay dotted lines for significant bins (per hemisphere)
        if bin_results:
            # LEFT HEMISPHERE dotted lines (black)
            if f'{roi}_lh' in bin_results:
                roi_bin_data_lh = bin_results[f'{roi}_lh'].get('bin_results', None)
                if roi_bin_data_lh and len(df_roi_lh) > 0:
                    bin_centers_lh = np.array(roi_bin_data_lh.get('bin_centers', []))
                    directions_lh = roi_bin_data_lh.get('directions', [])
                    
                    # Draw dotted lines for significant bins at appropriate heights
                    for bc, direc in zip(bin_centers_lh, directions_lh):
                        if 'High' in direc:
                            y_pos = lh_high_y_pos
                        elif 'Low' in direc:
                            y_pos = lh_low_y_pos
                        else:
                            continue  # Skip non-significant bins
                        
                        ax.plot([bc, bc + 1], [y_pos, y_pos], 
                               color='black', linestyle=':', linewidth=2.5, 
                               alpha=0.7, zorder=15)
            
            # RIGHT HEMISPHERE dotted lines (red)
            if f'{roi}_rh' in bin_results:
                roi_bin_data_rh = bin_results[f'{roi}_rh'].get('bin_results', None)
                if roi_bin_data_rh and len(df_roi_rh) > 0:
                    bin_centers_rh = np.array(roi_bin_data_rh.get('bin_centers', []))
                    directions_rh = roi_bin_data_rh.get('directions', [])
                    
                    # Draw dotted lines for significant bins at appropriate heights
                    for bc, direc in zip(bin_centers_rh, directions_rh):
                        if 'High' in direc:
                            y_pos = rh_high_y_pos
                        elif 'Low' in direc:
                            y_pos = rh_low_y_pos
                        else:
                            continue  # Skip non-significant bins
                        
                        ax.plot([bc, bc + 1], [y_pos, y_pos], 
                               color='red', linestyle=':', linewidth=2.5, 
                               alpha=0.7, zorder=15)
        
        # Add KL divergence info if available
        if kl_divergences:
            text_lines = []
            for pair, (kl_val, p_val) in kl_divergences.items():
                if roi in pair:
                    if p_val is not None:
                        sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                        text_lines.append(f'{pair}: KL={kl_val:.3f} ({sig_str})')
                    else:
                        text_lines.append(f'{pair}: KL={kl_val:.3f}')
            
            if text_lines:
                ax.text(0.02, 0.02, '\n'.join(text_lines),
                       transform=ax.transAxes,
                       fontsize=7, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Group GLM analysis (N={len(subject_ids)} subjects, LH=gray/black, RH=pink/red)', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✅ Figure saved: {output_path}")
    
    return fig


def create_analysis_summary_table(
    group_df: pd.DataFrame,
    roi_names: List[str],
    lme_results: Optional[Dict] = None,
    bayesian_results: Optional[object] = None,
    ecc_perm_results: Optional[Dict] = None,
    subject_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create comprehensive analysis summary table.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_names : List[str]
        List of ROI names
    lme_results : Optional[Dict]
        Results from fit_mixed_effects_model()
    bayesian_results : Optional[object]
        Results from fit_hierarchical_kl_model()
    ecc_perm_results : Optional[Dict]
        Results from eccentricity_permutation_test()
    subject_ids : Optional[List[str]]
        List of subject IDs
        
    Returns
    -------
    pd.DataFrame
        Summary table with all analyses
    """
    summary_rows = []
    
    for roi in roi_names:
        # Process LEFT HEMISPHERE
        df_roi_lh = group_df[(group_df['roi'] == roi) & (group_df['hemisphere'] == 'lh')]
        
        if len(df_roi_lh) > 0:
            row_lh = {
                'ROI': f'{roi}_LH',
                'N_vertices': len(df_roi_lh),
                #'N_subjects': df_roi_lh['subject_id'].nunique(),
                #'Mean_z': df_roi_lh['t_value'].mean(),
                #'SD_z': df_roi_lh['t_value'].std()
            }
            
            # KL divergence test results for LH
            if ecc_perm_results and f'{roi}_lh' in ecc_perm_results:
                ecc_perm = ecc_perm_results[f'{roi}_lh']
                row_lh['KL_observed'] = ecc_perm.get('observed_kl', np.nan)
                #row_lh['KL_p_value'] = ecc_perm.get('p_value', np.nan)
                row_lh['KL_null'] = ecc_perm.get('null_mean', np.nan)
                #row_lh['KL_null_std'] = ecc_perm.get('null_std', np.nan)
                
                # Per-bin directional test results
                bin_results = ecc_perm.get('bin_results', None)
                if bin_results and bin_results.get('directions'):
                    bin_centers = bin_results.get('bin_centers', [])
                    directions = bin_results.get('directions', [])
                    
                    # Separate bins by direction
                    bins_high = []
                    bins_low = []
                    bins_ns = []
                    
                    for bin_center, direction in zip(bin_centers, directions):
                        if 'High' in direction:
                            bins_high.append(bin_center)
                        elif 'Low' in direction:
                            bins_low.append(bin_center)
                        else:
                            bins_ns.append(bin_center)
                    
                    # Format as eccentricity ranges
                    def format_bin_range(bins):
                        if not bins:
                            return ''
                        bins = sorted(bins)
                        # Create ranges from consecutive bins (assuming 1° bins)
                        ranges = []
                        start = bins[0]
                        prev = bins[0]
                        for b in bins[1:]:
                            if b - prev > 1.5:  # Gap detected
                                ranges.append(f"{start:.0f}-{prev+1:.0f}°" if start != prev else f"{start:.0f}-{start+1:.0f}°")
                                start = b
                            prev = b
                        ranges.append(f"{start:.0f}-{prev+1:.0f}°" if start != prev else f"{start:.0f}-{start+1:.0f}°")
                        return ', '.join(ranges)
                    
                    row_lh['bins >'] = format_bin_range(bins_high)
                    row_lh['bins <'] = format_bin_range(bins_low)
                    #row_lh['Bins_ns'] = format_bin_range(bins_ns)
                else:
                    row_lh['bins >'] = ''
                    row_lh['bins <'] = ''
                    #row_lh['Bins_ns'] = ''
            else:
                row_lh['KL observed'] = np.nan
                #row_lh['KL_p_value'] = np.nan
                row_lh['KL null'] = np.nan
                #row_lh['KL_null_std'] = np.nan
                row_lh['bins >'] = ''
                row_lh['bins <'] = ''
                #row_lh['Bins_ns'] = ''
            
            summary_rows.append(row_lh)
        
        # Process RIGHT HEMISPHERE
        df_roi_rh = group_df[(group_df['roi'] == roi) & (group_df['hemisphere'] == 'rh')]
        
        if len(df_roi_rh) > 0:
            row_rh = {
                'ROI': f'{roi}_RH',
                'N_vertices': len(df_roi_rh),
                #'N_subjects': df_roi_rh['subject_id'].nunique(),
                'Mean_z': df_roi_rh['t_value'].mean(),
                #'SD_z': df_roi_rh['t_value'].std()
            }
            
            # KL divergence test results for RH
            if ecc_perm_results and f'{roi}_rh' in ecc_perm_results:
                ecc_perm = ecc_perm_results[f'{roi}_rh']
                row_rh['KL observed'] = ecc_perm.get('observed_kl', np.nan)
                #row_rh['KL_p_value'] = ecc_perm.get('p_value', np.nan)
                row_rh['KL null'] = ecc_perm.get('null_mean', np.nan)
                #row_rh['KL_null_std'] = ecc_perm.get('null_std', np.nan)
                
                # Per-bin directional test results
                bin_results = ecc_perm.get('bin_results', None)
                if bin_results and bin_results.get('directions'):
                    bin_centers = bin_results.get('bin_centers', [])
                    directions = bin_results.get('directions', [])
                    
                    # Separate bins by direction
                    bins_high = []
                    bins_low = []
                    bins_ns = []
                    
                    for bin_center, direction in zip(bin_centers, directions):
                        if 'High' in direction:
                            bins_high.append(bin_center)
                        elif 'Low' in direction:
                            bins_low.append(bin_center)
                        else:
                            bins_ns.append(bin_center)
                    
                    # Format as eccentricity ranges
                    def format_bin_range(bins):
                        if not bins:
                            return ''
                        bins = sorted(bins)
                        # Create ranges from consecutive bins (assuming 1° bins)
                        ranges = []
                        start = bins[0]
                        prev = bins[0]
                        for b in bins[1:]:
                            if b - prev > 1.5:  # Gap detected
                                ranges.append(f"{start:.0f}-{prev+1:.0f}°" if start != prev else f"{start:.0f}-{start+1:.0f}°")
                                start = b
                            prev = b
                        ranges.append(f"{start:.0f}-{prev+1:.0f}°" if start != prev else f"{start:.0f}-{start+1:.0f}°")
                        return ', '.join(ranges)
                    
                    row_rh['bins >'] = format_bin_range(bins_high)
                    row_rh['bins <'] = format_bin_range(bins_low)
                    #row_rh['Bins_ns'] = format_bin_range(bins_ns)
                else:
                    row_rh['bins >'] = ''
                    row_rh['bins <'] = ''
                    #row_rh['Bins_ns'] = ''
            else:
                row_rh['KL observed'] = np.nan
                #row_rh['KL_p_value'] = np.nan
                row_rh['KL null'] = np.nan
                #row_rh['KL_null_std'] = np.nan
                row_rh['bins >'] = ''
                row_rh['bins <'] = ''
                #row_rh['Bins_ns'] = ''
            
            summary_rows.append(row_rh)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add significance markers for KL divergence p-value
    if 'KL_p_value' in summary_df.columns:
        summary_df['KL_sig'] = summary_df['KL_p_value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' if not np.isnan(p) else ''
        )
    
    if 'LME_slope_p' in summary_df.columns:
        summary_df['LME_sig'] = summary_df['LME_slope_p'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        )
    
    return summary_df


def compute_tail_sign_per_roi(
    group_df: pd.DataFrame,
    threshold: float = 0.0
) -> Dict[str, Dict[str, float]]:
    """
    Compute tail sign metric: proportion of positive vs negative t-values per ROI.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    threshold : float, optional
        Threshold for considering a t-value significant (default: 0.0)
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict with ROI -> {'positive': float, 'negative': float, 'ratio': float}
    """
    results = {}
    
    for roi in group_df['roi'].unique():
        df_roi = group_df[group_df['roi'] == roi]
        
        n_positive = np.sum(df_roi['t_value'] > threshold)
        n_negative = np.sum(df_roi['t_value'] < -threshold)
        total = len(df_roi)
        
        ratio = n_positive / (n_negative + 1e-10)  # Avoid division by zero
        
        results[roi] = {
            'positive': n_positive / total,
            'negative': n_negative / total,
            'ratio': ratio
        }
    
    return results


def tail_sign_monte_carlo_null(
    group_df: pd.DataFrame,
    roi_name: str,
    n_permutations: int = 10000,
    threshold: float = 0.0,
    verbose: bool = True
) -> float:
    """
    Test if tail sign ratio differs from 1.0 (equal pos/neg) using permutation.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_name : str
        ROI to test
    n_permutations : int, optional
        Number of permutations (default: 10000)
    threshold : float, optional
        Threshold for t-values (default: 0.0)
    verbose : bool, optional
        Print progress (default: True)
        
    Returns
    -------
    float
        P-value for two-tailed test against ratio=1.0
    """
    df_roi = group_df[group_df['roi'] == roi_name].copy()
    
    # Observed ratio
    observed = compute_tail_sign_per_roi(df_roi, threshold)
    observed_ratio = observed[roi_name]['ratio']
    
    if verbose:
        print(f"Observed ratio ({roi_name}): {observed_ratio:.3f}")
    
    # Permutation test
    null_ratios = []
    
    for i in range(n_permutations):
        # Randomly flip signs
        shuffled_df = df_roi.copy()
        flip_mask = np.random.rand(len(shuffled_df)) < 0.5
        shuffled_df.loc[flip_mask, 't_value'] *= -1
        
        null_result = compute_tail_sign_per_roi(shuffled_df, threshold)
        null_ratios.append(null_result[roi_name]['ratio'])
    
    null_ratios = np.array(null_ratios)
    
    # Two-tailed p-value
    p_value = np.min([
        np.mean(null_ratios >= observed_ratio),
        np.mean(null_ratios <= observed_ratio)
    ]) * 2
    
    if verbose:
        print(f"P-value: {p_value:.4f}")
    
    # Memory cleanup: delete large arrays created during permutation testing
    try:
        del null_ratios
        gc.collect()
    except NameError:
        pass
    
    return p_value


def compute_effect_size(
    group_df: pd.DataFrame,
    roi_pair: Tuple[str, str],
    metric: str = 'cohen_d'
) -> float:
    """
    Compute effect size for difference between ROIs.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        DataFrame from aggregate_subjects_roi_data()
    roi_pair : Tuple[str, str]
        Pair of ROIs to compare
    metric : str, optional
        Effect size metric: 'cohen_d' or 'hedges_g' (default: 'cohen_d')
        
    Returns
    -------
    float
        Effect size value
    """
    roi1, roi2 = roi_pair
    
    df1 = group_df[group_df['roi'] == roi1]['t_value'].values
    df2 = group_df[group_df['roi'] == roi2]['t_value'].values
    
    mean1, mean2 = np.mean(df1), np.mean(df2)
    std1, std2 = np.std(df1, ddof=1), np.std(df2, ddof=1)
    n1, n2 = len(df1), len(df2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    cohen_d = (mean1 - mean2) / pooled_std
    
    if metric == 'hedges_g':
        # Hedges' g correction for small samples
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        return cohen_d * correction
    
    return cohen_d


# Utility functions for backward compatibility
def load_glm_contrast(subject_id: str, task: str, contrast: str, 
                     data_root: Path, hemi: str = 'lh') -> np.ndarray:
    """Load GLM contrast t-stat map for a single subject."""
    glm_path = (data_root / f'sub-{subject_id}' / 'ses-01' / 'func' / 
               f'task-{task}_combined' / f'{contrast}_stat.gii')
    gii = nib.load(str(glm_path))
    hemi_idx = 0 if hemi == 'lh' else 1
    return gii.darrays[hemi_idx].data


def load_benson_atlas(subject_id: str, benson_dir: Path, 
                     hemi: str = 'lh') -> Tuple[np.ndarray, np.ndarray]:
    """Load Benson V1-V3 ROI labels and eccentricity map."""
    roi_path = benson_dir / f'sub-{subject_id}_ses-01_iso' / 'surf' / f'{hemi}.benson14_varea.mgz'
    ecc_path = benson_dir / f'sub-{subject_id}_ses-01_iso' / 'surf' / f'{hemi}.benson14_eccen.mgz'
    
    roi_labels = nib.load(str(roi_path)).get_fdata().squeeze()
    eccentricity = nib.load(str(ecc_path)).get_fdata().squeeze()
    
    return roi_labels, eccentricity


if __name__ == '__main__':
    print("GLM Group Analysis Functions Module")
    print(f"PyMC available: {PYMC_AVAILABLE}")
    print(f"Statsmodels available: {STATSMODELS_AVAILABLE}")
    print(f"Neuropythy available: {ny is not None}")
