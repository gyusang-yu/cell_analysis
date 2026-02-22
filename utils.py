import os
import random
import numpy as np
import tifffile
import pandas as pd
import argparse
from skimage import measure, morphology
from cellpose import models, utils

# ------------------------------
# Argument Parsing
# ------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond', type=str, required=True, help='Experimental condition')
    parser.add_argument('--rep', type=int, required=False, help='Repetition index')
    parser.add_argument('--p', nargs='+', type=int, required=False, help='Page indices to process')
    parser.add_argument('--img', type=str, required=True, help='Base folder name for image stack, e.g., 5D_MAY09')
    return parser.parse_args()

# ------------------------------
# Path Setup (for labeling only)
# ------------------------------
def setup_paths(condition, rep, img_folder):
    output_dir = f'./processed/{img_folder}_{condition}'
    label_dir = os.path.join(output_dir, f'labels_{rep}')
    outline_dir = os.path.join(output_dir, f'outline_{rep}')
    result_dir = os.path.join(output_dir, 'result')

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(outline_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    return label_dir, outline_dir, result_dir

# ------------------------------
# Load Image Stacks
# ------------------------------
def load_image_stacks(condition, rep, img_folder):
    base_path = f'./{img_folder}'
    inos_path = sorted([f for f in os.listdir(base_path) if f"{condition}_{rep}_ch00" in f])[0]
    phalloidin_path = sorted([f for f in os.listdir(base_path) if f"{condition}_{rep}_ch01" in f])[0]
    dapi_path = sorted([f for f in os.listdir(base_path) if f"{condition}_{rep}_ch02" in f])[0]

    inos_stack = tifffile.imread(os.path.join(base_path, inos_path))
    ph_stack = tifffile.imread(os.path.join(base_path, phalloidin_path))
    dapi_stack = tifffile.imread(os.path.join(base_path, dapi_path))
    return inos_stack, ph_stack, dapi_stack

# ------------------------------
# Label Filtering
# ------------------------------
def remove_labels_on_edges(label_img):
    label_img = np.copy(label_img)
    edge_labels = np.unique(np.concatenate([
        label_img[0, :], label_img[-1, :],
        label_img[:, 0], label_img[:, -1]
    ]))
    for label in edge_labels:
        if label != 0:
            label_img[label_img == label] = 0
    return label_img

def remove_labels_out_of_size_range(label_img, min_size=500, max_size=20000):
    label_img = np.copy(label_img)
    props = measure.regionprops(label_img)
    for prop in props:
        if prop.area <= 0 or prop.area < min_size or prop.area > max_size:
            label_img[label_img == prop.label] = 0
    return label_img



# ------------------------------
# Cellpose Execution
# ------------------------------
def run_cellpose_and_filter(image, model, diameter, i, second_channel, label_dir, label_name):
    image_input = np.stack([image, second_channel], axis=-1)
    channels = [0, 1]
    masks, _, _ = model.eval(image_input, diameter=diameter, channels=channels,
                             cellprob_threshold=0.0, flow_threshold=0.4)
    mask_no_edges = remove_labels_on_edges(masks)
    if not (image == second_channel).all() :
        mask_filtered = remove_labels_out_of_size_range(mask_no_edges)
    else : 
        mask_filtered = mask_no_edges
    label_save_path = os.path.join(label_dir, f'label_{label_name}_page{i}.tif')
    tifffile.imwrite(label_save_path, mask_filtered.astype(np.uint16))
    print(f"Saved label image to {label_save_path}")

# ------------------------------
# Generate Outline Mask
# ------------------------------
def generate_outline(mask, outline_path):
    outlines = utils.outlines_list(mask)
    outline_mask = np.zeros_like(mask, dtype=bool)
    for outline in outlines:
        coords = outline.astype(int)
        outline_mask[coords[:, 1], coords[:, 0]] = True
    tifffile.imwrite(outline_path, outline_mask.astype(np.uint8) * 255)
    return outline_mask

# ------------------------------
# Generate DAPI Mask by thresholding intensity
# ------------------------------

def generate_dapi_label(dapi_img, model, diameter, i, label_dir):
    """
    Step-by-step DAPI label generator:
    - Threshold DAPI signal
    - Remove small objects (area < 50)
    """
    masks, _, _ = model.eval(dapi_img, diameter=diameter, channels=None,
                             cellprob_threshold=0.0, flow_threshold=0.4)
    mask_no_edges = remove_labels_on_edges(masks)

    label_save_path = os.path.join(label_dir, f'label_dapi_page{i}.tif')
    tifffile.imwrite(label_save_path, mask_no_edges.astype(np.uint16))
    print(f"Saved label image to {label_save_path}")

# ------------------------------
# Match label IDs between DAPI and iNOS
# ------------------------------

def match_label_ids(i, raw_label_dir, inos_label, dapi_label, overlap_threshold=0.5, max_labels=None):
    """
    Matches iNOS and DAPI labels based on spatial overlap, and limits the number
    of matched labels to 'max_labels' by filtering only successful matches.

    Args:
        i (int): Page index, used in output filenames.
        raw_label_dir (str): Directory to save the matched label files.
        inos_label (ndarray): Label image for iNOS.
        dapi_label (ndarray): Label image for DAPI.
        overlap_threshold (float): Minimum fraction of DAPI label area that must overlap with iNOS label.
        max_labels (int or None): Desired number of successful matched labels. None for no limit.
    """
    matched_dapi_label = np.zeros_like(dapi_label, dtype=np.uint16)
    matched_inos_label = np.zeros_like(inos_label, dtype=np.uint16)

    dapi_props = measure.regionprops(dapi_label)
    inos_props = measure.regionprops(inos_label)

    # Precompute binary masks for each DAPI label
    dapi_masks = {p.label: dapi_label == p.label for p in dapi_props if p.label != 0}

    # Get and shuffle iNOS IDs
    inos_ids = [p.label for p in inos_props if p.label != 0]
    random.seed(42)
    random.shuffle(inos_ids)

    current_id = 1  # count matched (successful) labels, initialization

    for inos_id in inos_ids:
        if max_labels is not None and current_id > max_labels:
            break

        inos_mask = inos_label == inos_id

        overlap_ids, counts = np.unique(dapi_label[inos_mask], return_counts=True)
        valid = overlap_ids != 0
        overlap_ids = overlap_ids[valid]
        counts = counts[valid]

        dapi_union_mask = np.zeros_like(dapi_label, dtype=bool)
        matched_any = False

        for dapi_id, count in zip(overlap_ids, counts):
            dapi_area = np.sum(dapi_masks[dapi_id])
            overlap_fraction = count / dapi_area

            if overlap_fraction >= overlap_threshold:
                dapi_union_mask |= dapi_masks[dapi_id]
                matched_any = True

        if matched_any:
            matched_dapi_label[dapi_union_mask] = current_id
            matched_inos_label[inos_mask] = current_id
            current_id += 1

    # Save output
    dapi_label_save_path = os.path.join(raw_label_dir, f'label_dapi_final_page{i}.tif')
    inos_label_save_path = os.path.join(raw_label_dir, f'label_inos_final_page{i}.tif')
    tifffile.imwrite(dapi_label_save_path, matched_dapi_label.astype(np.uint16))
    tifffile.imwrite(inos_label_save_path, matched_inos_label.astype(np.uint16))
    print(f"[Page {i}] Saved {current_id - 1} matched labels (target: {max_labels}) to {raw_label_dir}")


# ------------------------------
# Measurement Extraction
# ------------------------------


def extract_measurements(condition, rep, inos_label, ph_img, inos_img, dapi_label, image_index, global_label_counter):
    measurements = []

    props_nuc = {p.label: p for p in measure.regionprops(dapi_label, intensity_image=inos_img)}
    props_tf  = {p.label: p for p in measure.regionprops(inos_label, intensity_image=inos_img)}
    props_ph  = {p.label: p for p in measure.regionprops(inos_label, intensity_image=ph_img)}

    # Match by label ID
    common_labels = sorted(set(props_nuc.keys()) & set(props_tf.keys()) & set(props_ph.keys()))

    for label_id in common_labels:
        prop_nuc = props_nuc[label_id]
        prop_tf = props_tf[label_id]
        prop_ph = props_ph[label_id]
        
        # if prop_tf.area < 500:
        #     continue

        cyto_area = prop_tf.area - prop_nuc.area
        if cyto_area <=10 : 
            continue

        cyto_total = prop_tf.mean_intensity * prop_tf.area - prop_nuc.mean_intensity * prop_nuc.area
        if cyto_total < 0 :
            continue

        TF_cytoplasm_mean = cyto_total / cyto_area if cyto_area > 0 else 0.0

        TF_nc_ratio_mean = prop_nuc.mean_intensity / TF_cytoplasm_mean if TF_cytoplasm_mean > 0 else 0.0
        TF_nc_ratio_total = (prop_nuc.mean_intensity * prop_nuc.area) / cyto_total if cyto_total > 0 else 0.0

        record = {
            'Condition': condition,
            'Rep_index' : rep,
            'Image_Index': image_index,
            'Label_ID': global_label_counter,
            'Phalloidin_Mean_Intensity': prop_ph.mean_intensity,
            'TF_Mean_Intensity': prop_tf.mean_intensity,
            'Cell_Area': prop_tf.area,
            'nucleus_Area' : prop_nuc.area,
            'area_difference' : prop_tf.area - prop_nuc.area,
            'Phalloidin_Total_Intensity': prop_ph.mean_intensity * prop_ph.area,
            'TF_Total_Intensity': prop_tf.mean_intensity * prop_tf.area,

            'TF_nuclear_mean': prop_nuc.mean_intensity,
            'TF_nuclear_total': prop_nuc.mean_intensity * prop_nuc.area,
            'TF_cytoplasm_mean': TF_cytoplasm_mean,
            'TF_cytoplasm_total': cyto_total,
            'TF_nc_ratio_mean': TF_nc_ratio_mean,
            'TF_nc_ratio_total': TF_nc_ratio_total,
        }

        measurements.append(record)
        global_label_counter += 1

    return measurements, global_label_counter
