import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import neuropythy as ny
import scipy.stats as stats
from nilearn import signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, kendalltau
from scipy.spatial import procrustes
from pingouin import circ_corrcl

from prfpy.stimulus import CFStimulus
from prfpy.model import CFGaussianModel
from prfpy.fit import CFFitter 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge


def optimize_connfield_dfree(
    gf,
    r2_threshold=0.05,
    sigma_bounds=[(0.1, 20.0)],
    method='L-BFGS-B',
    tol=0.001,
    verbose=True
):
    """Optimize connective field parameters using BFGS starting from grid search results.
    Only optimizes sigma while keeping vertex center fixed (standard CF approach).
    
    Args:
        gf (CFFitter): The CFFitter object after grid_fit has been run
        r2_threshold (float): R² threshold for optimization (default 0.05)
        verbose (bool): Print progress information
    
    Returns:
        dict: Contains optimized parameters:
            - 'vertex_indices': Fixed vertex centers from grid search
            - 'sigma': Optimized sigma values
            - 'beta': Optimized beta (amplitude) values  
            - 'baseline': Optimized baseline values
            - 'r2': r2 values for optimized fits
    """
    from scipy.optimize import minimize
    from tqdm import tqdm
    
    ## Get grid search results
    best_vertex_indices = gf.gridsearch_params[:, 0].astype(int)
    best_sigmas = gf.gridsearch_params[:, 1]
    r2_mask = gf.gridsearch_r2 >= r2_threshold
    #r2_mask = gf.gridsearch_params[:, 4] >= r2_threshold

    
    n_sites = gf.data.shape[0]
    n_optimize = r2_mask.sum()
    
    if verbose:
        print(f"Optimizing {n_optimize} of {n_sites} sites (r2 >= {r2_threshold})")
    
    # Get distance matrix and source data from stimulus
    distance_matrix = gf.model.stimulus.distance_matrix
    source_data = gf.model.stimulus.design_matrix  # n_vertices x n_timepoints
    
    def optimize_single_voxel(voxel_idx):
        """Optimize sigma for a single voxel"""
        vertex_idx = int(best_vertex_indices[voxel_idx])
        sigma_init = best_sigmas[voxel_idx]
        target_ts = gf.data[voxel_idx]
        
        # Get distances from this vertex center to all source vertices
        distances = distance_matrix[vertex_idx, :]  # Shape: (n_vertices,)
        
        def objective(sigma):
            """Objective function: minimize negative correlation"""
            # Calculate CF weights
            weights = np.exp(-(distances**2 / (2 * sigma[0]**2)))  # Shape: (n_vertices,)
            
            # Create CF timecourse - weighted sum across vertices
            # source_data is (n_vertices, n_timepoints)
            # weights is (n_vertices,)
            cf_ts = np.dot(weights, source_data)  # Shape: (n_timepoints,)
            
            # Calculate correlation
            r = np.corrcoef(target_ts, cf_ts)[0, 1]
            
            # Return negative correlation (to minimize)
            return -r if not np.isnan(r) else 1.0
        
        # Optimize
        result = minimize(
            objective,
            x0=[sigma_init],
            bounds=sigma_bounds,  
            method=method,
            tol=tol
        )
        
        # Calculate final r2
        sigma_opt = result.x[0]
        weights_opt = np.exp(-(distances**2 / (2 * sigma_opt**2)))
        cf_ts_opt = np.dot(weights_opt, source_data)
        r_opt = np.corrcoef(target_ts, cf_ts_opt)[0, 1]
        r2_opt = r_opt**2 if not np.isnan(r_opt) else 0.0
        
        # Estimate beta and baseline using least squares
        X = np.column_stack([cf_ts_opt, np.ones_like(cf_ts_opt)])
        params = np.linalg.lstsq(X, target_ts, rcond=None)[0]
        beta_opt = params[0]
        baseline_opt = params[1]
        
        return {
            'vertex_idx': vertex_idx,
            'sigma': sigma_opt,
            'beta': beta_opt,
            'baseline': baseline_opt,
            'r2': r2_opt
        }
    
    # Run optimization with progress bar
    sites_to_optimize = np.where(r2_mask)[0]
    results = []
    
    for voxel_idx in tqdm(sites_to_optimize, desc="Optimizing CFs", disable=not verbose):
        results.append(optimize_single_voxel(voxel_idx))
    
    # Initialize output arrays with grid search values
    optimized_params = {
        'vertex_indices': best_vertex_indices.copy(),
        'sigma': best_sigmas.copy(),
        'beta': gf.gridsearch_params[:, 2].copy(),
        'baseline': gf.gridsearch_params[:, 3].copy(),
        'r2': gf.gridsearch_r2.copy()
    }
    
    # Update with optimized values
    for i, voxel_idx in enumerate(sites_to_optimize):
        optimized_params['sigma'][voxel_idx] = results[i]['sigma']
        optimized_params['beta'][voxel_idx] = results[i]['beta']
        optimized_params['baseline'][voxel_idx] = results[i]['baseline']
        optimized_params['r2'][voxel_idx] = results[i]['r2']
    
    if verbose:
        sigma_improvement = optimized_params['sigma'][r2_mask] - best_sigmas[r2_mask]
        r2_improvement = optimized_params['r2'][r2_mask] - gf.gridsearch_r2[r2_mask]
        print(f"\nOptimization complete!")
        print(f"Mean sigma change: {np.mean(np.abs(sigma_improvement)):.3f} mm")
        print(f"Mean r2 improvement: {np.mean(r2_improvement):.4f}")
        print(f"Optimized sigma range: {optimized_params['sigma'][r2_mask].min():.2f} - {optimized_params['sigma'][r2_mask].max():.2f} mm")
    
    return optimized_params


def optimize_connfield_gdescent(
    gf,
    r2_threshold=0.05,
    sigma_bounds=(0.1, 20.0),
    learning_rate=0.01,
    max_iterations=1000,
    convergence_threshold=1e-4,
    batch_size=128,
    verbose=True
):
    """GPU-accelerated PARALLEL CF optimization using TensorFlow."""
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    
    ## Get grid search results
    best_vertex_indices = gf.gridsearch_params[:, 0].astype(np.int32)
    best_sigmas = gf.gridsearch_params[:, 1].astype(np.float32)
    r2_mask = gf.gridsearch_r2 >= r2_threshold
    
    n_sites = gf.data.shape[0]
    n_optimize = r2_mask.sum()
    
    if verbose:
        print(f"\n{'='*60}")
        print("TENSORFLOW GPU PARALLEL OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Optimizing {n_optimize} of {n_sites} sites (r2 >= {r2_threshold})")
        print(f"Batch size: {batch_size}")
        if gpus:
            print(f"GPU: {gpus[0].name}")
        else:
            print("⚠ WARNING: No GPU detected!")
    
    # Get data
    distance_matrix = gf.model.stimulus.distance_matrix.astype(np.float32)
    source_data = gf.model.stimulus.design_matrix.astype(np.float32)
    target_data = gf.data.astype(np.float32)
    
    # Convert to TF constants (stays on GPU)
    distance_matrix_tf = tf.constant(distance_matrix, dtype=tf.float32)
    source_data_tf = tf.constant(source_data, dtype=tf.float32)
    
    # Convert bounds to TF constants
    sigma_min = tf.constant(sigma_bounds[0], dtype=tf.float32)
    sigma_max = tf.constant(sigma_bounds[1], dtype=tf.float32)
    
    # Initialize outputs
    optimized_params = {
        'vertex_indices': best_vertex_indices.copy(),
        'sigma': best_sigmas.copy(),
        'beta': gf.gridsearch_params[:, 2].copy(),
        'baseline': gf.gridsearch_params[:, 3].copy(),
        'r2': gf.gridsearch_r2.copy()
    }
    
    sites_to_optimize = np.where(r2_mask)[0]
    n_batches = int(np.ceil(n_optimize / batch_size))
    
    @tf.function
    def optimize_batch_parallel(log_sigmas, vertex_indices, distances_batch, target_batch, 
                                source_data_matrix, sigma_min_val, sigma_max_val):
        """Optimize entire batch in parallel on GPU - ALL VARIABLES AS ARGUMENTS"""
        # Transform sigmas (all at once)
        sigmas = tf.exp(log_sigmas)
        sigmas = tf.clip_by_value(sigmas, sigma_min_val, sigma_max_val)
        
        # Compute weights for all sites: [batch_size, n_source_vertices]
        sigmas_expanded = tf.expand_dims(sigmas, axis=1)  # [batch_size, 1]
        weights = tf.exp(-(distances_batch ** 2) / (2 * sigmas_expanded ** 2))
        
        # Compute CF timecourses for all sites: [batch_size, n_timepoints]
        cf_timecourses = tf.matmul(weights, source_data_matrix)
        
        # Normalize CF timecourses
        cf_mean = tf.reduce_mean(cf_timecourses, axis=1, keepdims=True)
        cf_std = tf.math.reduce_std(cf_timecourses, axis=1, keepdims=True) + 1e-8
        cf_normalized = (cf_timecourses - cf_mean) / cf_std
        
        # Normalize target timecourses
        target_mean = tf.reduce_mean(target_batch, axis=1, keepdims=True)
        target_std = tf.math.reduce_std(target_batch, axis=1, keepdims=True) + 1e-8
        target_normalized = (target_batch - target_mean) / target_std
        
        # Compute correlations
        correlations = tf.reduce_mean(cf_normalized * target_normalized, axis=1)
        
        # Return negative correlation as loss
        loss = -tf.reduce_mean(correlations)
        
        return loss, sigmas
    
    # Process in batches
    for batch_idx in tqdm(range(n_batches), desc="GPU batches", disable=not verbose):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_optimize)
        batch_voxel_indices = sites_to_optimize[start_idx:end_idx]
        batch_size_actual = len(batch_voxel_indices)
        
        # Get batch data
        batch_vertices = best_vertex_indices[batch_voxel_indices]
        batch_init_sigmas = best_sigmas[batch_voxel_indices]
        
        # Initialize log sigmas for batch
        log_sigmas_init = np.log(np.clip(batch_init_sigmas, sigma_bounds[0] + 0.01, sigma_bounds[1] - 0.01))
        log_sigmas = tf.Variable(log_sigmas_init, dtype=tf.float32, trainable=True)
        
        # Get distances for all sites in batch: [batch_size, n_source_vertices]
        distances_batch = tf.gather(distance_matrix_tf, batch_vertices, axis=0)
        
        # Get target data for batch: [batch_size, n_timepoints]
        target_batch = tf.constant(target_data[batch_voxel_indices], dtype=tf.float32)
        
        # Create single optimizer for entire batch
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        prev_loss = float('inf')
        no_improvement = 0
        
        # Optimize all sites in batch simultaneously
        for iteration in range(max_iterations):
            with tf.GradientTape() as tape:
                # Pass ALL variables as arguments (no closures!)
                loss, sigmas_opt = optimize_batch_parallel(
                    log_sigmas, batch_vertices, distances_batch, target_batch,
                    source_data_tf, sigma_min, sigma_max  # Pass bounds as arguments
                )
            
            # Compute gradients for all sigmas at once
            gradients = tape.gradient(loss, [log_sigmas])
            optimizer.apply_gradients(zip(gradients, [log_sigmas]))
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                current_loss = float(loss.numpy())
                if abs(prev_loss - current_loss) < convergence_threshold:
                    no_improvement += 1
                    if no_improvement >= 3:
                        break
                else:
                    no_improvement = 0
                prev_loss = current_loss
        
        # Extract final optimized sigmas
        _, sigmas_final = optimize_batch_parallel(
            log_sigmas, batch_vertices, distances_batch, target_batch,
            source_data_tf, sigma_min, sigma_max
        )
        sigmas_opt_np = sigmas_final.numpy()
        
        # Compute final parameters (this part is CPU-bound but fast)
        for i, voxel_idx in enumerate(batch_voxel_indices):
            vertex_idx = int(batch_vertices[i])
            sigma_opt = float(sigmas_opt_np[i])
            
            # Compute final CF timecourse
            distances_np = distance_matrix[vertex_idx, :]
            weights_opt = np.exp(-(distances_np ** 2) / (2 * sigma_opt ** 2))
            cf_ts_opt = np.dot(weights_opt, source_data)
            
            # Fit beta and baseline
            target_ts_np = target_data[voxel_idx]
            X = np.column_stack([cf_ts_opt, np.ones_like(cf_ts_opt)])
            params = np.linalg.lstsq(X, target_ts_np, rcond=None)[0]
            beta_opt = float(params[0])
            baseline_opt = float(params[1])
            
            # Compute R²
            r_opt = np.corrcoef(target_ts_np, cf_ts_opt)[0, 1]
            r2_opt = float(r_opt**2) if not np.isnan(r_opt) else 0.0
            
            # Store results
            optimized_params['sigma'][voxel_idx] = sigma_opt
            optimized_params['beta'][voxel_idx] = beta_opt
            optimized_params['baseline'][voxel_idx] = baseline_opt
            optimized_params['r2'][voxel_idx] = r2_opt
    
    tf.keras.backend.clear_session()
    
    if verbose:
        sigma_improvement = optimized_params['sigma'][r2_mask] - best_sigmas[r2_mask]
        r2_improvement = optimized_params['r2'][r2_mask] - gf.gridsearch_r2[r2_mask]
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nSigma changes:")
        print(f"  Mean |Δσ|: {np.mean(np.abs(sigma_improvement)):.3f} mm")
        print(f"  Max |Δσ|: {np.max(np.abs(sigma_improvement)):.3f} mm")
        print(f"\nR² improvements:")
        print(f"  Mean ΔR²: {np.mean(r2_improvement):.4f}")
        print(f"  Sites improved: {(r2_improvement > 0).sum()} ({100*(r2_improvement > 0).sum()/n_optimize:.1f}%)")
        print(f"{'='*60}")
    
    return optimized_params


def optimize_connfield_joint(
    gf,
    r2_threshold=0.05,
    sigma_bounds=(0.1, 20.0),
    max_outer_iterations=3,
    max_inner_iterations=300,
    search_radius=10.0,
    batch_size=256,
    learning_rate=0.01,
    verbose=True
):
    """GPU-parallelized alternating position-sigma optimization.
    
    Processes multiple sites in parallel on GPU for both:
    1. Position search (vectorized evaluation)
    2. Sigma optimization (batched gradient descent)
    
    Returns:
        dict: Contains optimized parameters and cycle-wise statistics
    """
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm
    
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    
    ## Get grid search results
    best_vertex_indices = gf.gridsearch_params[:, 0].astype(np.int32)
    best_sigmas = gf.gridsearch_params[:, 1].astype(np.float32)
    r2_mask = gf.gridsearch_r2 >= r2_threshold
    
    n_sites = gf.data.shape[0]
    n_optimize = r2_mask.sum()
    
    if verbose:
        print(f"\n{'='*60}")
        print("GPU-PARALLELIZED ALTERNATING OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Optimizing {n_optimize} of {n_sites} sites")
        print(f"Cycles: {max_outer_iterations}, Batch: {batch_size}")
        if gpus:
            print(f"GPU: {gpus[0].name}")
        else:
            print("⚠ WARNING: No GPU!")
    
    # Get data
    distance_matrix = gf.model.stimulus.distance_matrix.astype(np.float32)
    source_data = gf.model.stimulus.design_matrix.astype(np.float32)
    target_data = gf.data.astype(np.float32)
    
    # Move to GPU
    distance_matrix_tf = tf.constant(distance_matrix, dtype=tf.float32)
    source_data_tf = tf.constant(source_data, dtype=tf.float32)
    target_data_tf = tf.constant(target_data, dtype=tf.float32)
    
    # Convert bounds to TF constants
    sigma_min = tf.constant(sigma_bounds[0], dtype=tf.float32)
    sigma_max = tf.constant(sigma_bounds[1], dtype=tf.float32)
    
    # Pre-compute neighbor lists (CPU)
    n_source_vertices = distance_matrix.shape[0]
    max_neighbors = 0
    neighbor_lists = []
    for vertex_idx in range(n_source_vertices):
        neighbors = np.where(distance_matrix[vertex_idx, :] <= search_radius)[0]
        neighbor_lists.append(neighbors)
        max_neighbors = max(max_neighbors, len(neighbors))
    
    # Pad neighbor lists for GPU (all same length)
    neighbor_array = np.full((n_source_vertices, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_source_vertices, dtype=np.int32)
    for i, neighbors in enumerate(neighbor_lists):
        n = len(neighbors)
        neighbor_array[i, :n] = neighbors
        neighbor_counts[i] = n
    
    neighbor_array_tf = tf.constant(neighbor_array, dtype=tf.int32)
    neighbor_counts_tf = tf.constant(neighbor_counts, dtype=tf.int32)
    
    if verbose:
        print(f"Max neighbors: {max_neighbors}, Search radius: {search_radius} mm")
    
    # Initialize current state
    current_vertices = best_vertex_indices.copy()
    current_sigmas = best_sigmas.copy()
    current_r2 = gf.gridsearch_r2.copy()
    
    sites_to_optimize = np.where(r2_mask)[0]
    n_batches = int(np.ceil(n_optimize / batch_size))
    
    # Initialize cycle statistics storage
    cycle_stats = {
        'cycle': [],
        'n_improved': [],
        'percent_improved': [],
        'n_position_changes': [],
        'percent_position_changes': [],
        'mean_sigma_change': [],
        'median_sigma_change': [],
        'max_sigma_change': [],
        'mean_r2': [],
        'mean_r2_improvement': [],
        'median_r2_improvement': []
    }
    
    @tf.function
    def evaluate_positions_batch(vertex_candidates, sigmas, distances_batch, target_batch,
                                 source_data_matrix, sigma_min_val, sigma_max_val):
        """Evaluate all candidate positions for a batch of sites in parallel."""
        sigmas_expanded = tf.reshape(sigmas, [-1, 1, 1])
        weights = tf.exp(-(distances_batch ** 2) / (2 * sigmas_expanded ** 2))
        cf_timeseries = tf.einsum('bcv,vt->bct', weights, source_data_matrix)
        
        cf_mean = tf.reduce_mean(cf_timeseries, axis=2, keepdims=True)
        cf_std = tf.math.reduce_std(cf_timeseries, axis=2, keepdims=True) + 1e-8
        cf_normalized = (cf_timeseries - cf_mean) / cf_std
        
        target_expanded = tf.expand_dims(target_batch, axis=1)
        target_mean = tf.reduce_mean(target_expanded, axis=2, keepdims=True)
        target_std = tf.math.reduce_std(target_expanded, axis=2, keepdims=True) + 1e-8
        target_normalized = (target_expanded - target_mean) / target_std
        
        correlations = tf.reduce_mean(cf_normalized * target_normalized, axis=2)
        return correlations
    
    @tf.function
    def optimize_sigma_batch(log_sigmas, vertex_indices, distances_batch, target_batch,
                            source_data_matrix, sigma_min_val, sigma_max_val):
        """Optimize sigma for a batch of sites (fixed positions) in parallel."""
        sigmas = tf.exp(log_sigmas)
        sigmas = tf.clip_by_value(sigmas, sigma_min_val, sigma_max_val)
        
        sigmas_expanded = tf.expand_dims(sigmas, axis=1)
        weights = tf.exp(-(distances_batch ** 2) / (2 * sigmas_expanded ** 2))
        cf_timeseries = tf.matmul(weights, source_data_matrix)
        
        cf_mean = tf.reduce_mean(cf_timeseries, axis=1, keepdims=True)
        cf_std = tf.math.reduce_std(cf_timeseries, axis=1, keepdims=True) + 1e-8
        cf_normalized = (cf_timeseries - cf_mean) / cf_std
        
        target_mean = tf.reduce_mean(target_batch, axis=1, keepdims=True)
        target_std = tf.math.reduce_std(target_batch, axis=1, keepdims=True) + 1e-8
        target_normalized = (target_batch - target_mean) / target_std
        
        correlations = tf.reduce_mean(cf_normalized * target_normalized, axis=1)
        loss = -tf.reduce_mean(correlations)
        
        return loss, sigmas
    
    # Store initial state for comparison
    initial_vertices = current_vertices.copy()
    initial_sigmas = current_sigmas.copy()
    initial_r2 = current_r2.copy()
    
    # ALTERNATING OPTIMIZATION CYCLES
    for outer_iter in range(max_outer_iterations):
        if verbose:
            print(f"\n--- Cycle {outer_iter + 1}/{max_outer_iterations} ---")
        
        # Store state at beginning of cycle
        cycle_start_vertices = current_vertices.copy()
        cycle_start_sigmas = current_sigmas.copy()
        cycle_start_r2 = current_r2.copy()
        
        cycle_improvements = 0
        
        # Process in batches
        for batch_idx in tqdm(range(n_batches), 
                             desc=f"Cycle {outer_iter+1}",
                             disable=not verbose):
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_optimize)
            batch_voxel_indices = sites_to_optimize[start_idx:end_idx]
            batch_size_actual = len(batch_voxel_indices)
            
            batch_vertices = current_vertices[batch_voxel_indices]
            batch_sigmas = current_sigmas[batch_voxel_indices]
            batch_r2 = current_r2[batch_voxel_indices]
            batch_target = target_data[batch_voxel_indices]
            
            # STEP 1: OPTIMIZE POSITION
            batch_neighbors = neighbor_array[batch_vertices]
            batch_n_neighbors = neighbor_counts[batch_vertices]
            
            batch_neighbors_tf = tf.constant(batch_neighbors, dtype=tf.int32)
            batch_sigmas_tf = tf.constant(batch_sigmas, dtype=tf.float32)
            batch_target_tf = tf.constant(batch_target, dtype=tf.float32)
            
            distances_batch = tf.gather(distance_matrix_tf, batch_neighbors_tf, axis=0)
            valid_mask = batch_neighbors_tf >= 0
            
            correlations = evaluate_positions_batch(
                batch_neighbors_tf, batch_sigmas_tf, distances_batch, batch_target_tf,
                source_data_tf, sigma_min, sigma_max
            )
            
            correlations = tf.where(valid_mask, correlations, -1.0)
            best_candidate_indices = tf.argmax(correlations, axis=1, output_type=tf.int32)
            best_candidate_indices_np = best_candidate_indices.numpy()
            
            new_vertices = np.array([
                batch_neighbors[i, best_candidate_indices_np[i]] 
                for i in range(batch_size_actual)
            ])
            
            # STEP 2: OPTIMIZE SIGMA
            log_sigmas_init = np.log(np.clip(batch_sigmas, sigma_bounds[0] + 0.01, sigma_bounds[1] - 0.01))
            log_sigmas = tf.Variable(log_sigmas_init, dtype=tf.float32, trainable=True)
            
            new_vertices_tf = tf.constant(new_vertices, dtype=tf.int32)
            distances_batch_sigma = tf.gather(distance_matrix_tf, new_vertices_tf, axis=0)
            
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
            
            prev_loss = float('inf')
            no_improvement = 0
            
            for inner_iter in range(max_inner_iterations):
                with tf.GradientTape() as tape:
                    loss, _ = optimize_sigma_batch(
                        log_sigmas, new_vertices_tf, distances_batch_sigma, batch_target_tf,
                        source_data_tf, sigma_min, sigma_max
                    )
                
                gradients = tape.gradient(loss, [log_sigmas])
                optimizer.apply_gradients(zip(gradients, [log_sigmas]))
                
                if inner_iter % 10 == 0:
                    current_loss = float(loss.numpy())
                    if abs(prev_loss - current_loss) < 1e-4:
                        no_improvement += 1
                        if no_improvement >= 3:
                            break
                    else:
                        no_improvement = 0
                    prev_loss = current_loss
            
            _, sigmas_opt_tf = optimize_sigma_batch(
                log_sigmas, new_vertices_tf, distances_batch_sigma, batch_target_tf,
                source_data_tf, sigma_min, sigma_max
            )
            new_sigmas = sigmas_opt_tf.numpy()
            
            # STEP 3: Compute final R² and update if improved
            for i in range(batch_size_actual):
                voxel_idx = batch_voxel_indices[i]
                vertex_idx = int(new_vertices[i])
                sigma = float(new_sigmas[i])
                target_ts = batch_target[i]
                
                distances_np = distance_matrix[vertex_idx, :]
                weights = np.exp(-(distances_np ** 2) / (2 * sigma ** 2))
                cf_ts = np.dot(weights, source_data)
                
                r = np.corrcoef(target_ts, cf_ts)[0, 1]
                r2 = r**2 if not np.isnan(r) else 0.0
                
                if r2 > batch_r2[i]:
                    current_vertices[voxel_idx] = vertex_idx
                    current_sigmas[voxel_idx] = sigma
                    current_r2[voxel_idx] = r2
                    cycle_improvements += 1
        
        # Calculate cycle statistics
        position_changes = (current_vertices[r2_mask] != cycle_start_vertices[r2_mask])
        n_position_changes = position_changes.sum()
        
        sigma_changes = current_sigmas[r2_mask] - cycle_start_sigmas[r2_mask]
        abs_sigma_changes = np.abs(sigma_changes)
        
        r2_improvements = current_r2[r2_mask] - cycle_start_r2[r2_mask]
        
        # Store cycle statistics
        cycle_stats['cycle'].append(outer_iter + 1)
        cycle_stats['n_improved'].append(int(cycle_improvements))
        cycle_stats['percent_improved'].append(float(100 * cycle_improvements / n_optimize))
        cycle_stats['n_position_changes'].append(int(n_position_changes))
        cycle_stats['percent_position_changes'].append(float(100 * n_position_changes / n_optimize))
        cycle_stats['mean_sigma_change'].append(float(abs_sigma_changes.mean()))
        cycle_stats['median_sigma_change'].append(float(np.median(abs_sigma_changes)))
        cycle_stats['max_sigma_change'].append(float(abs_sigma_changes.max()))
        cycle_stats['mean_r2'].append(float(current_r2[r2_mask].mean()))
        cycle_stats['mean_r2_improvement'].append(float(r2_improvements.mean()))
        cycle_stats['median_r2_improvement'].append(float(np.median(r2_improvements)))
        
        if verbose:
            print(f"  Improved: {cycle_improvements}/{n_optimize} sites ({100*cycle_improvements/n_optimize:.1f}%)")
            print(f"  Position changes: {n_position_changes} sites ({100*n_position_changes/n_optimize:.1f}%)")
            print(f"  Mean |Δσ|: {abs_sigma_changes.mean():.3f} mm")
            print(f"  Mean R²: {current_r2[r2_mask].mean():.4f}")
            print(f"  Mean ΔR²: {r2_improvements.mean():.4f}")
        
        # Stop if converged
        if cycle_improvements == 0 and outer_iter > 0:
            if verbose:
                print(f"\nConverged after {outer_iter + 1} cycles")
            break
    
    # Compute final beta and baseline
    optimized_params = {
        'vertex_indices': current_vertices,
        'sigma': current_sigmas,
        'beta': gf.gridsearch_params[:, 2].copy(),
        'baseline': gf.gridsearch_params[:, 3].copy(),
        'r2': current_r2,
        'cycle_statistics': cycle_stats  # NEW: Add cycle-wise statistics
    }
    
    # Recompute beta/baseline for optimized sites
    for voxel_idx in sites_to_optimize:
        vertex_idx = int(current_vertices[voxel_idx])
        sigma = float(current_sigmas[voxel_idx])
        target_ts = target_data[voxel_idx]
        
        distances_np = distance_matrix[vertex_idx, :]
        weights = np.exp(-(distances_np ** 2) / (2 * sigma ** 2))
        cf_ts = np.dot(weights, source_data)
        
        X = np.column_stack([cf_ts, np.ones_like(cf_ts)])
        params = np.linalg.lstsq(X, target_ts, rcond=None)[0]
        
        optimized_params['beta'][voxel_idx] = float(params[0])
        optimized_params['baseline'][voxel_idx] = float(params[1])
    
    tf.keras.backend.clear_session()
    
    if verbose:
        print(f"\n{'='*60}")
        print("FINAL RESULTS (from initial grid search)")
        print(f"{'='*60}")
        
        position_changed = (optimized_params['vertex_indices'][r2_mask] != initial_vertices[r2_mask])
        sigma_change = np.abs(optimized_params['sigma'][r2_mask] - initial_sigmas[r2_mask])
        r2_improvement = optimized_params['r2'][r2_mask] - initial_r2[r2_mask]
        
        print(f"\nTotal position changes:")
        print(f"  Sites moved: {position_changed.sum()} ({100*position_changed.sum()/n_optimize:.1f}%)")
        
        print(f"\nTotal sigma changes:")
        print(f"  Mean |Δσ|: {sigma_change.mean():.3f} mm")
        print(f"  Median |Δσ|: {np.median(sigma_change):.3f} mm")
        print(f"  Max |Δσ|: {sigma_change.max():.3f} mm")
        
        print(f"\nTotal R² improvements:")
        print(f"  Mean ΔR²: {r2_improvement.mean():.4f}")
        print(f"  Median ΔR²: {np.median(r2_improvement):.4f}")
        print(f"  Sites improved: {(r2_improvement > 0).sum()} ({100*(r2_improvement > 0).sum()/n_optimize:.1f}%)")
        print(f"  Mean final R²: {optimized_params['r2'][r2_mask].mean():.4f}")
        print(f"{'='*60}")
    
    return optimized_params

## Joint optimzation cycles

def plot_convergence_summary_table(cycle_stats):
    """Create a summary table of convergence statistics."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create DataFrame
    df = pd.DataFrame(cycle_stats)
    
    # Add computed columns
    df['convergence_rate'] = df['percent_improved'].diff().fillna(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.4)), dpi=150)
    ax.axis('tight')
    ax.axis('off')
    
    # Format DataFrame for display
    display_df = df[[
        'cycle', 
        'n_improved',
        'percent_improved',
        'n_position_changes',
        'percent_position_changes',
        'mean_sigma_change',
        'mean_r2',
        'mean_r2_improvement'
    ]].copy()
    
    display_df.columns = [
        'Cycle',
        'N Improved',
        '% Improved',
        'N Pos. changes',
        '% Pos. changes',
        'Mean |Δσ| (mm)',
        'Mean R²',
        'Mean ΔR²'
    ]
    
    # Format numeric columns
    display_df['% Improved'] = display_df['% Improved'].map('{:.2f}%'.format)
    display_df['% Pos. changes'] = display_df['% Pos. changes'].map('{:.2f}%'.format)
    display_df['Mean |Δσ| (mm)'] = display_df['Mean |Δσ| (mm)'].map('{:.3f}'.format)
    display_df['Mean R²'] = display_df['Mean R²'].map('{:.4f}'.format)
    display_df['Mean ΔR²'] = display_df['Mean ΔR²'].map('{:.5f}'.format)
    
    # Create table
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Joint-optimization summary', 
             fontsize=14, fontweight='bold', pad=20)
    
    return fig

## Compare two optimization strategies

def compare_cf_results(
    results_A,
    results_B,
    ecc_pRF,
    pol_pRF,
    target_roi_mask,
    flatmaps,
    colors_ecc,
    colors_polar,
    h='lh',
    r2_threshold=0.1,
    ecc_div=2,
    figsize=(16, 8),
    dpi=300
):
    """Compare connective field results from grid search vs optimization.
    
    Args:
        results_A (dict): Dictionary with keys 'centers', 'sigma', 'r2'
        results_B (dict): Dictionary with keys 'centers', 'sigma', 'r2'
        ecc_pRF (np.array): Eccentricity values for source vertices
        pol_pRF (np.array): Polar angle values for source vertices
        sub_target_mask (np.array): Boolean mask for target vertices
        flatmaps (dict): Dictionary of flatmap objects
        colors_ecc (dict): Eccentricity color palette
        colors_polar (dict): Polar angle color palette
        h (str): Hemisphere ('lh' or 'rh')
        r2_threshold (float): R² threshold for visualization
        figsize (tuple): Figure size
        dpi (int): Figure DPI
    
    Returns:
        matplotlib.figure.Figure: Comparison figure
    """
    
    # Create figure with two rows (grid search top, optimized bottom)
    # Add extra column for row labels
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.15])
    
    CF_ecc_A = ecc_pRF[results_A['centers'].astype(int)]
    CF_pol_A = pol_pRF[results_A['centers'].astype(int)]
    sigma_A = results_A['sigma']
    r2_A = results_A['r2']
    
    CF_ecc_B = ecc_pRF[results_B['centers'].astype(int)]
    CF_pol_B = pol_pRF[results_B['centers'].astype(int)]
    sigma_B = results_B['sigma']
    r2_B = results_B['r2']
    
    # Plot both rows
    for row_idx, (CF_ecc, CF_polar, sigma, r2, row_label) in enumerate([
        (CF_ecc_A, CF_pol_A, sigma_A, r2_A, results_A['opType']),
        (CF_ecc_B, CF_pol_B, sigma_B, r2_B, results_B['opType'])
    ]):
        # Create maps
        ecc_map = np.full(target_roi_mask.shape[0], np.nan)
        ecc_map[target_roi_mask] = CF_ecc
        
        polar_map = np.full(target_roi_mask.shape[0], np.nan)
        polar_map[target_roi_mask] = CF_polar
        
        sigma_map = np.full(target_roi_mask.shape[0], np.nan)
        sigma_map[target_roi_mask] = sigma
        
        r2_map = np.full(target_roi_mask.shape[0], np.nan)
        r2_map[target_roi_mask] = r2
        
        # Create mask for R² threshold
        threshold_mask = r2_map >= r2_threshold
        
        # Get axes for this row
        left_ax = fig.add_subplot(gs[row_idx, 0])
        left_middle_ax = fig.add_subplot(gs[row_idx, 1])
        right_middle_ax = fig.add_subplot(gs[row_idx, 2])
        right_ax = fig.add_subplot(gs[row_idx, 3])
        label_ax = fig.add_subplot(gs[row_idx, 4])
        
        # Eccentricity plot
        ny.cortex_plot(
            flatmaps[h],
            axes=left_ax,
            color=ecc_map,
            cmap=colors_ecc['matplotlib_cmap'],
            mask=threshold_mask,
            vmin=np.nanmin(ecc_map),
            vmax=np.nanmax(ecc_map)/ecc_div,
        )
        left_ax.set_aspect('equal')
        if row_idx == 0:  # Only add column titles to top row
            left_ax.set_title('CF eccentricity', pad=8, fontsize=20) 
        left_ax.axis('off')
        
        # Polar angle plot
        ny.cortex_plot(
            flatmaps[h],
            axes=left_middle_ax,
            color=polar_map,
            cmap=colors_polar['matplotlib_cmap'],
            mask=threshold_mask,
        )
        left_middle_ax.set_aspect('equal')
        if row_idx == 0:
            left_middle_ax.set_title('CF polar angle', pad=8, fontsize=20) 
        left_middle_ax.axis('off')
        
        # CF Size plot
        size_vmin = np.nanmin(sigma_map)
        size_vmax = np.nanmax(sigma_map)
        size_cmap = plt.cm.jet
        
        ny.cortex_plot(
            flatmaps[h],
            axes=right_middle_ax,
            color=sigma_map,
            cmap=size_cmap,
            mask=threshold_mask,
            vmin=size_vmin,
            vmax=size_vmax,
        )
        right_middle_ax.set_aspect('equal')
        if row_idx == 0:
            right_middle_ax.set_title('CF size', pad=8, fontsize=20) 
        right_middle_ax.axis('off')
        
        # Variance explained plot
        varex_cmap = plt.cm.inferno
        ny.cortex_plot(
            flatmaps[h],
            axes=right_ax,
            color=r2_map,
            cmap=varex_cmap,
            mask=threshold_mask,
            vmin=0,
            vmax=1,
        )
        right_ax.set_aspect('equal')
        if row_idx == 0:
            right_ax.set_title('Variance explained', pad=8, fontsize=20) 
        right_ax.axis('off')
        
        # Add vertical row label on the right
        label_ax.axis('off')
        label_ax.text(0.5, 0.5, row_label, 
                     rotation=90, 
                     fontsize=14, 
                     fontweight='bold',
                     ha='center', 
                     va='center',
                     transform=label_ax.transAxes)
        
        # Add colorbars only to bottom row
        if row_idx == 1:
            # Eccentricity inset
            ecc_inset = inset_axes(left_ax, width="50%", height="50%",
                                   loc="lower right", borderpad=-6)
            ecc_inset.set_aspect('equal')
            ecc_inset.set_xlim(-1.5, 1.5)
            ecc_inset.set_ylim(-1.5, 1.5)
            ecc_inset.text(0.5, -0.05, r'CF center $\rho\ (\mathit{deg})$',
                          ha='center', va='top', fontsize=14,
                          transform=ecc_inset.transAxes)
            ecc_inset.set_axis_off()
            
            num_ecc_colors = len(colors_ecc["hex"])
            for i, color in enumerate(colors_ecc["hex"]):
                inner_r = i / num_ecc_colors
                outer_r = (i + 1) / num_ecc_colors
                ring = Wedge((0, 0), outer_r, 0, 360,
                           width=outer_r - inner_r, color=color)
                ecc_inset.add_patch(ring)
            
            # Polar angle inset
            polar_inset = inset_axes(left_middle_ax, width="40%", height="40%",
                                    loc="lower right", borderpad=-6)
            polar_inset.set_aspect('equal')
            polar_inset.set_axis_off()
            polar_inset.pie([1]*len(colors_polar["hex"]),
                          colors=colors_polar["hex"],
                          startangle=180, counterclock=False)
            polar_inset.text(0.5, -0.05, r'CF center $\theta\ (\mathit{rad})$',
                           ha='center', va='top', fontsize=14,
                           transform=polar_inset.transAxes)
            
            # CF Size colorbar
            sigma_rect_ax = inset_axes(right_middle_ax, width="30%", height="10%",
                                      loc="lower right", borderpad=-3)
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            gradient = np.vstack((gradient, gradient))
            sigma_rect_ax.imshow(gradient, aspect='auto', cmap=size_cmap,
                               extent=[0, 1, 0, 1])
            sigma_rect_ax.text(0, -0.3, f'{size_vmin:.2f}',
                             ha='left', va='top', fontsize=10)
            sigma_rect_ax.text(1, -0.3, f'{size_vmax:.2f}',
                             ha='right', va='top', fontsize=10)
            sigma_rect_ax.text(0.5, 1.3, r'CF $\sigma\ (\mathit{mm})$',
                             ha='center', va='bottom', fontsize=14,
                             transform=sigma_rect_ax.transAxes)
            sigma_rect_ax.axis('off')
            
            # Variance explained colorbar
            varex_rect_ax = inset_axes(right_ax, width="30%", height="10%",
                                      loc="lower right", borderpad=-3)
            varex_rect_ax.imshow(gradient, aspect='auto', cmap=varex_cmap,
                               extent=[0, 1, 0, 1])
            varex_rect_ax.text(0, -0.3, '0', ha='left', va='top', fontsize=12)
            varex_rect_ax.text(1, -0.3, '1', ha='right', va='top', fontsize=12)
            varex_rect_ax.text(0.5, 1.3, r'$\mathit{r}\!{}^2$',
                             ha='center', va='bottom', fontsize=14,
                             transform=varex_rect_ax.transAxes)
            varex_rect_ax.axis('off')
    
    plt.tight_layout()
    
    # Print comparison statistics
    print("\n" + "="*60)
    print("Optimization approach comparison")
    print("="*60)
    
    mask = r2_A > r2_threshold
    print(f"\nR² threshold: {r2_threshold}")
    print(f"Sites above threshold: {mask.sum()}")
    
    print(f"\n")
    print(results_A['opType'])
    print(f"Sigma range: {sigma_A[mask].min():.2f} - {sigma_A[mask].max():.2f} mm")
    print(f"Mean sigma: {sigma_A[mask].mean():.2f} mm")
    print(f"Mean R²: {r2_A[mask].mean():.4f}")
    
    print(f"\n")
    print(results_B['opType'])
    print(f"Sigma range: {sigma_B[mask].min():.2f} - {sigma_B[mask].max():.2f} mm")
    print(f"Mean sigma: {sigma_B[mask].mean():.2f} mm")
    print(f"Mean R²: {r2_B[mask].mean():.4f}")
    
    print("\n--- Improvements ---")
    sigma_change = np.abs(sigma_B[mask] - sigma_A[mask])
    r2_improvement = r2_B[mask] - r2_A[mask]
    print(f"Mean |Δσ|: {sigma_change.mean():.3f} mm")
    print(f"Mean ΔR²: {r2_improvement.mean():.4f}")
    print(f"Median ΔR²: {np.median(r2_improvement):.4f}")
    print(f"Sites with improved R²: {(r2_improvement > 0).sum()} ({100*(r2_improvement > 0).sum()/mask.sum():.1f}%)")
    print("="*60)
    
    return fig

