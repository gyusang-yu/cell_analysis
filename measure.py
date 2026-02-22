import os
import re
import tifffile
import numpy as np
import pandas as pd
from glob import glob
from utils import extract_measurements, load_image_stacks, generate_outline, setup_paths

if __name__ == '__main__':
    label_root = './processed/labels_final'
    inos_labels = sorted(glob(os.path.join(label_root, '*inos*.tif')))
    merged_df = pd.DataFrame()
    global_label_counter = 1

    for inos_path in inos_labels:
        basename = os.path.basename(inos_path).replace('.tif', '')
        # pattern = r'^(?P<img_folder>.+)_(?P<condition>MVG\d+(?:\.\d+)?|Ctrl)_(?P<rep>\d+)_inos_final_page(?P<page>\d+)$'
        pattern = r'^(?P<img_folder>.+)_(?P<condition>.+)_(?P<rep>[^_]+)_inos_final_page(?P<page>\d+)$'
        match = re.match(pattern, basename)
        if not match:
            print(f"Filename does not match expected pattern: {basename}")
            continue

        img_folder = match.group('img_folder')
        condition = match.group('condition')
        rep = int(match.group('rep'))
        page_idx = int(match.group('page'))

        dapi_path = inos_path.replace('inos', 'dapi')

        if not os.path.exists(dapi_path):
            print(f"Missing DAPI label for {inos_path}, expected at {dapi_path}")
            continue

        try:
            # Now load the image stacks based on parsed metadata
            inos_stack, ph_stack, dapi_stack = load_image_stacks(condition, rep, img_folder)
            ph_img = ph_stack[page_idx]
            inos_img = inos_stack[page_idx]
            inos_label = tifffile.imread(inos_path)
            dapi_label = tifffile.imread(dapi_path)
        except IndexError as e:
            print(f"Index error at page {page_idx}: {e}")
            continue

        # Set up directories for this combination
        _, outline_dir, measure_dir = setup_paths(condition, rep, img_folder)

        outline_path = os.path.join(outline_dir, f'outline_image_page{page_idx}.tif')
        if not os.path.exists(outline_path):
            outline_mask = generate_outline(inos_label, outline_path)
            inos_label[outline_mask] = 0
            tifffile.imwrite(inos_path, inos_label.astype(np.uint16))
            print(f"Saved outline and updated {inos_path}")
        else:
            print(f"Outline exists: {outline_path}")

        # Run measurement
        records, global_label_counter = extract_measurements(
            condition=condition,
            rep = rep,
            inos_label=inos_label,
            ph_img=ph_img,
            inos_img=inos_img,
            dapi_label=dapi_label,
            image_index=page_idx,
            global_label_counter=global_label_counter
        )
        df = pd.DataFrame(records)


        # Save per-page CSV
        os.makedirs(measure_dir, exist_ok=True)
        csv_path = os.path.join(measure_dir, f'measurement_{condition}_{rep}_page{page_idx}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved measurement to {csv_path}")

        # Merge into full dataframe
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    # Save global merged CSV
    os.makedirs('processed', exist_ok=True)
    merged_csv_path = os.path.join('processed', f'measurement_merged.csv')
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Saved merged results to {merged_csv_path}")
