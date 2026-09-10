import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.spatial
from tqdm import tqdm
import time
import os
import argparse

def process_tile(tile_img, x_offset, y_offset, tile_bounds, min_area=3, max_area=100, top_hat_size=15, method="peak", min_dist=5, threshold=None, invert=False):
    """
    Process a single tile to detect pillars and compute their properties.
    tile_bounds is a tuple of (x_min, x_max, y_min, y_max) defining the unique zone
    for detections in global coordinates (to prevent duplicate counts at boundaries).
    """
    # Ensure tile_img is float32
    if tile_img.dtype != np.float32:
        tile_img = tile_img.astype(np.float32)
        
    if invert:
        # For float32 normalized to [0, 1]
        tile_img = 1.0 - tile_img

    # 1. Background subtraction using Morphological Top-Hat filter
    # This isolates small bright spots (pillars) on a varying background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (top_hat_size, top_hat_size))
    # cv2.morphologyEx works with float32 directly
    tophat = cv2.morphologyEx(tile_img, cv2.MORPH_TOPHAT, kernel)
    
    # 2. Noise reduction (increased to 5x5 to better filter high-frequency noise)
    blurred = cv2.GaussianBlur(tophat, (5, 5), 0)
    
    if method == "blob":
        # 3. Binarization (Otsu's threshold is excellent here since background is zeroed out)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Connected Components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        if num_labels <= 1:
            return [] # Only background found
            
        # Get component statistics
        areas = stats[1:, cv2.CC_STAT_AREA] # Skip label 0 (background)
        
        # Filter component IDs by area
        valid_indices = np.where((areas >= min_area) & (areas <= max_area))[0] + 1
        
        if len(valid_indices) == 0:
            return []
            
        # Extract centroids
        centroids_valid = centroids[valid_indices]
        
        # Compute global coordinates of centroids
        x_global = centroids_valid[:, 0] + x_offset
        y_global = centroids_valid[:, 1] + y_offset
        
        # Filter detections to keep only those within the unique boundary of this tile
        x_min, x_max, y_min, y_max = tile_bounds
        in_bounds = (x_global >= x_min) & (x_global < x_max) & (y_global >= y_min) & (y_global < y_max)
        
        valid_indices = valid_indices[in_bounds]
        x_global = x_global[in_bounds]
        y_global = y_global[in_bounds]
        
        if len(valid_indices) == 0:
            return []
            
        # 5. Extract intensity metrics
        # Calculate exact mask average intensity in the original image using scipy.ndimage
        mean_intensities = scipy.ndimage.mean(tile_img, labels=labels, index=valid_indices)
        
        # Calculate 3x3 box average around integer centroids (fast local neighborhood)
        box_blur = cv2.blur(tile_img, (3, 3))
        xi_local = np.round(centroids[valid_indices, 0]).astype(np.int32)
        yi_local = np.round(centroids[valid_indices, 1]).astype(np.int32)
        
        # Clip indices to prevent out of bounds
        h_tile, w_tile = tile_img.shape
        xi_local = np.clip(xi_local, 0, w_tile - 1)
        yi_local = np.clip(yi_local, 0, h_tile - 1)
        
        box_intensities = box_blur[yi_local, xi_local]
        
        # Gather results
        tile_results = []
        areas_valid = stats[valid_indices, cv2.CC_STAT_AREA]
        
        for i in range(len(valid_indices)):
            tile_results.append({
                'x': x_global[i],
                'y': y_global[i],
                'area': float(areas_valid[i]),
                'mean_intensity': float(mean_intensities[i]),
                'box_intensity_3x3': float(box_intensities[i])
            })
            
        return tile_results

    elif method == "peak":
        # 3. Local maximum detection using dilation (excellent for dense close-packed structures)
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * min_dist + 1, 2 * min_dist + 1))
        dilated = cv2.dilate(blurred, kernel_dil)
        
        # Determine threshold using local background subtraction (highly robust to local illumination variations)
        # Compute local background using a box filter (size 21x21 is larger than a few pillars)
        bg_local = cv2.boxFilter(blurred, -1, (21, 21))
        local_contrast = blurred - bg_local
        
        # DYNAMIC THRESHOLDING FOR FLOAT32
        # Since image is [0, 1], absolute thresholds like 1.5 are invalid.
        # We use a percentile-based approach on the positive local contrast.
        if threshold is None:
            # Only consider positive contrast pixels to find a robust statistical threshold for peaks
            pos_contrast = local_contrast[local_contrast > 0]
            if len(pos_contrast) > 0:
                # 99.5th percentile of positive contrast is a robust dynamic threshold for sparse peaks
                thresh_val = np.percentile(pos_contrast, 99.5)
            else:
                thresh_val = 0.01 # Fallback
        else:
            # If user provides a threshold, treat it as a multiplier for standard deviation
            pos_contrast = local_contrast[local_contrast > 0]
            if len(pos_contrast) > 0:
                std_contrast = np.std(pos_contrast)
                thresh_val = threshold * std_contrast
            else:
                thresh_val = 0.01
                
        peaks_mask = (blurred == dilated) & (local_contrast >= thresh_val)
        
        # Resolve flat-top duplicate peaks using connected components
        num_peaks, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks_mask.astype(np.uint8))
        
        if num_peaks <= 1:
            return []
            
        # Centroids of the peak groups (skip label 0 which is background)
        xi_peak = centroids[1:, 0]
        yi_peak = centroids[1:, 1]
        
        # Round to nearest integer coordinates for local region analysis
        xi_int = np.round(xi_peak).astype(np.int32)
        yi_int = np.round(yi_peak).astype(np.int32)
        
        # Filter out peaks too close to tile borders to prevent out-of-bounds in sub-pixel centroid
        h_tile, w_tile = tile_img.shape
        margin = 2
        valid_peaks = (xi_int >= margin) & (xi_int < w_tile - margin) & (yi_int >= margin) & (yi_int < h_tile - margin)
        
        xi_int = xi_int[valid_peaks]
        yi_int = yi_int[valid_peaks]
        xi_peak = xi_peak[valid_peaks]
        yi_peak = yi_peak[valid_peaks]
        
        if len(xi_int) == 0:
            return []
            
        # Calculate sub-pixel centroids using center of mass in a 3x3 window around the peaks
        x_sub = []
        y_sub = []
        for i in range(len(xi_int)):
            px, py = xi_int[i], yi_int[i]
            # 3x3 window around peak
            W = blurred[py-1:py+2, px-1:px+2].astype(np.float32)
            sum_W = np.sum(W)
            if sum_W > 0:
                # Weighted average for X
                x_c = (W[0, 0]*(px-1) + W[0, 1]*px + W[0, 2]*(px+1) +
                       W[1, 0]*(px-1) + W[1, 1]*px + W[1, 2]*(px+1) +
                       W[2, 0]*(px-1) + W[2, 1]*px + W[2, 2]*(px+1)) / sum_W
                # Weighted average for Y
                y_c = (W[0, 0]*(py-1) + W[0, 1]*(py-1) + W[0, 2]*(py-1) +
                       W[1, 0]*py     + W[1, 1]*py     + W[1, 2]*py     +
                       W[2, 0]*(py+1) + W[2, 1]*(py+1) + W[2, 2]*(py+1)) / sum_W
                x_sub.append(x_c)
                y_sub.append(y_c)
            else:
                x_sub.append(float(xi_peak[i]))
                y_sub.append(float(yi_peak[i]))
                
        x_sub = np.array(x_sub)
        y_sub = np.array(y_sub)
        
        # Calculate global coordinates
        x_global = x_sub + x_offset
        y_global = y_sub + y_offset
        
        # Filter detections to keep only those within the unique boundary of this tile
        x_min, x_max, y_min, y_max = tile_bounds
        in_bounds = (x_global >= x_min) & (x_global < x_max) & (y_global >= y_min) & (y_global < y_max)
        
        x_global = x_global[in_bounds]
        y_global = y_global[in_bounds]
        xi_int = xi_int[in_bounds]
        yi_int = yi_int[in_bounds]
        
        if len(x_global) == 0:
            return []
            
        # Extract intensities (average in 3x3 area of original tile image)
        box_blur = cv2.blur(tile_img, (3, 3))
        box_intensities = box_blur[yi_int, xi_int]
        
        tile_results = []
        for i in range(len(x_global)):
            tile_results.append({
                'x': float(x_global[i]),
                'y': float(y_global[i]),
                'area': 9.0, # Nominal area for peak-based detection
                'mean_intensity': float(box_intensities[i]),
                'box_intensity_3x3': float(box_intensities[i])
            })
        return tile_results

def prune_close_peaks(df_results, min_dist=4):
    if df_results.empty:
        return df_results
    coords = df_results[['x', 'y']].values
    intensities = df_results['mean_intensity'].values
    tree = scipy.spatial.KDTree(coords)
    pairs = tree.query_pairs(r=min_dist)
    to_discard = set()
    for i, j in pairs:
        if i in to_discard or j in to_discard:
            continue
        if intensities[i] >= intensities[j]:
            to_discard.add(j)
        else:
            to_discard.add(i)
    df_pruned = df_results.drop(index=list(to_discard)).reset_index(drop=True)
    return df_pruned

def analyze_image(image_path, tile_size=2000, overlap=50, min_area=3, max_area=100, top_hat_size=15, method="peak", min_dist=5, threshold=None, invert=False):
    print(f"Loading image from {image_path}...")
    t0 = time.time()
    # Read preserving 16-bit depth (unicode path safe)
    try:
        # IMREAD_ANYDEPTH preserves 16-bit TIFFs instead of truncating to 8-bit
        img_raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    except Exception as e:
        img_raw = None
    if img_raw is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # LAYER A: 16-bit to float32 [0.0 - 1.0] normalization
    # Raw max can be arbitrary (e.g. 65535 or 4095). We scale based on robust percentiles or theoretical max.
    # Assuming 16-bit theoretical max (65535) for absolute consistency across images.
    img = img_raw.astype(np.float32) / 65535.0
    
    # Note: Global Flat-field Normalization (Median Scaling) is temporarily disabled 
    # to maintain strict absolute intensities for rigorous paired difference testing.
    
    height, width = img.shape
    print(f"Loaded image of dimensions {width}x{height} pixels ({width*height/1e6:.1f} MP). Converted to float32 [0.0 - 1.0]. Read in {time.time() - t0:.2f} seconds.")
    
    # Calculate grid steps
    step = tile_size - 2 * overlap
    x_steps = int(np.ceil(width / step))
    y_steps = int(np.ceil(height / step))
    
    print(f"Processing in a {x_steps}x{y_steps} grid of tiles (tile size: {tile_size}, overlap: {overlap})...")
    print(f"Detection Method: {method.upper()}")
    
    results = []
    
    # Loop over the tiles
    for r in range(y_steps):
        for c in range(x_steps):
            # Define unique zone for this tile (non-overlapping boundaries)
            x_min = c * step
            x_max = min((c + 1) * step, width)
            y_min = r * step
            y_max = min((r + 1) * step, height)
            
            # Define read zone including overlap
            x_start = max(0, x_min - overlap)
            x_end = min(width, x_max + overlap)
            y_start = max(0, y_min - overlap)
            y_end = min(height, y_max + overlap)
            
            # Crop the tile
            tile = img[y_start:y_end, x_start:x_end]
            
            # Process tile
            tile_results = process_tile(
                tile_img=tile,
                x_offset=x_start,
                y_offset=y_start,
                tile_bounds=(x_min, x_max, y_min, y_max),
                min_area=min_area,
                max_area=max_area,
                top_hat_size=top_hat_size,
                method=method,
                min_dist=min_dist,
                threshold=threshold,
                invert=invert
            )
            
            results.extend(tile_results)
            
    print(f"Detection completed. Total raw detections: {len(results):,}")
    
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(results, columns=["x", "y", "area", "mean_intensity", "box_intensity_3x3"])
    
    # Prune close peaks if using peak method
    if method == "peak" and not df_results.empty:
        print(f"Pruning overlapping peaks (distance < {min_dist} pixels)...")
        t_prune = time.time()
        # We prune using (min_dist - 0.5) to avoid pruning legitimate closely-packed hexagonal peaks
        # but discarding true double-peak duplicates
        prune_radius = max(2.5, min_dist - 0.5)
        df_results = prune_close_peaks(df_results, min_dist=prune_radius)
        print(f"Pruning completed in {time.time() - t_prune:.2f} seconds. Remaining pillars: {len(df_results):,}")
        
    df_results.insert(0, 'pillar_id', np.arange(len(df_results)))
    df_results["detection_method"] = method
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plasmon Dark-Field Image Analyzer")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="C:/Users/chuya/.gemini/antigravity/scratch/plasmon_analyzer/results.csv", help="Path to save output CSV")
    parser.add_argument("--tile-size", type=int, default=2000, help="Tile size for processing")
    parser.add_argument("--overlap", type=int, default=50, help="Tile overlap pixels")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum pixel area for a pillar (for 'blob' method)")
    parser.add_argument("--max-area", type=int, default=100, help="Maximum pixel area for a pillar (for 'blob' method)")
    parser.add_argument("--top-hat", type=int, default=15, help="Background top-hat filter size")
    
    # Added for HCP/dense peak method
    parser.add_argument("--method", type=str, default="peak", choices=["blob", "peak"], help="Detection method: 'blob' (connected components) or 'peak' (local maxima)")
    parser.add_argument("--min-dist", type=int, default=5, help="Minimum distance between peaks (for 'peak' method)")
    parser.add_argument("--threshold", type=float, default=None, help="Positive local-contrast SD multiplier (peak); omitted: 99.5th percentile")
    parser.add_argument("--invert", action="store_true", help="Invert image contrast to detect dark features on bright background")
    
    args = parser.parse_args()
    
    start_time = time.time()
    df = analyze_image(
        image_path=args.image,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_area=args.min_area,
        max_area=args.max_area,
        top_hat_size=args.top_hat,
        method=args.method,
        min_dist=args.min_dist,
        threshold=args.threshold,
        invert=args.invert
    )
    
    print("Saving results...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
