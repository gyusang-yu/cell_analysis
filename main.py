# from utils import get_args, setup_paths, run_cellpose_and_filter, load_image_stacks, generate_dapi_label, match_label_ids
from utils import *
import tifffile
import os
from glob import glob
from cellpose import models
import napari
import numpy as np

if __name__ == '__main__':
    args = get_args()
    condition = args.cond
    rep = args.rep
    page_filter = args.p
    img_folder = args.img

    inos_stack, ph_stack, dapi_stack = load_image_stacks(condition, rep, img_folder)
    raw_label_dir, _, _ = setup_paths(condition, rep, img_folder)
    os.makedirs(raw_label_dir, exist_ok=True)


    model = models.CellposeModel(model_type="cyto3", gpu=True)
    dapi_model = models.CellposeModel(model_type="nuclei", gpu=True)
    
    ph_sizemodel = models.SizeModel(model, pretrained_size=models.size_model_path("cyto3"))
    dapi_sizemodel = models.SizeModel(dapi_model, pretrained_size=models.size_model_path("nuclei"))

    if page_filter is None:
        print("No page filter provided. Launching image stack only in napari viewer...")
        viewer = napari.Viewer()
        viewer.add_image(ph_stack, name='Phalloidin', blending='translucent', contrast_limits=[0, 150], colormap='red')
        viewer.add_image(inos_stack, name='iNOS', blending='additive', contrast_limits=[0, 40], colormap='green')
        viewer.add_image(dapi_stack, name='DAPI', blending='additive', contrast_limits=[0, 60], colormap='blue')
        napari.run()
        exit()

    if page_filter == [-1]:
        page_filter = list(range(ph_stack.shape[0]))


    for i in page_filter:
        label_path = glob(os.path.join(raw_label_dir, f'label_*_page{i}.tif'))
        if label_path:
            print(f"Label for slice {i} already exists, skipping Cellpose segmentation.")
            continue

        ph_img = ph_stack[i]
        in_img = inos_stack[i]
        dapi_img = dapi_stack[i]
        
        ph_diam = ph_sizemodel.eval(ph_img)[0]
        dapi_diam = dapi_sizemodel.eval(dapi_img)[0]
        
        print(f"Estimated diameter for slice {i}: {ph_diam}")
        print(f"Estimated dapi diameter for slice {i}: {dapi_diam}")

        run_cellpose_and_filter(ph_img, model, ph_diam, i, second_channel=in_img,
                                label_dir=raw_label_dir, label_name="inos") ####################### change name here for pAb name
        inos_label = tifffile.imread(os.path.join(raw_label_dir, f'label_inos_page{i}.tif'))
        generate_dapi_label(dapi_img, dapi_model, dapi_diam, i, raw_label_dir)
        dapi_label = tifffile.imread(os.path.join(raw_label_dir, f'label_dapi_page{i}.tif'))

        match_label_ids(i, raw_label_dir, inos_label, dapi_label, overlap_threshold=0.8 , max_labels=120) ################## change max label # per images

    print("Labeling complete. Launching napari viewer...")

    viewer = napari.Viewer()
    viewer.add_image(ph_stack, name='Phalloidin', blending='translucent', contrast_limits=[0, 150], colormap='red')
    viewer.add_image(inos_stack, name='iNOS', blending='additive', contrast_limits=[0, 40], colormap='green')
    viewer.add_image(dapi_stack, name='DAPI', blending='additive', contrast_limits=[0, 60], colormap='blue')
    label_layers = []
    for i in page_filter:
        label_path = glob(os.path.join(raw_label_dir, f'label_*final_page{i}.tif'))
        for j in label_path:
            label_img = tifffile.imread(j)
            layer = viewer.add_labels(label_img, name=j)
            label_layers.append((i, layer))

    labels_final_path = f'./processed/labels_final'
    os.makedirs(labels_final_path, exist_ok=True)
    print(f"You can now manually edit labels and export to: {labels_final_path}/{img_folder}_{condition}_{rep}_{{index}}.tif")
    print("Press \'cmd+\\\' to save edited labels")

    @viewer.bind_key('Command-\\')
    def save_all_labels(viewer):
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                original_path = layer.name
                filename = os.path.basename(original_path)  # ex) label_inos_page1.tif
                if filename.startswith("label_"):
                    filename = filename.replace("label_", f"{img_folder}_{condition}_{rep}_")


                save_path = os.path.join(labels_final_path, filename)
                tifffile.imwrite(save_path, layer.data.astype(np.uint16))

                original_dir = os.path.dirname(original_path)
                backup_save_path = os.path.join(original_dir, filename.replace('label_', 'final_label_'))
                tifffile.imwrite(backup_save_path, layer.data.astype(np.uint16))
                print(f"Saved label: {save_path}")
    
    napari.run()
