import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from pydicom.pixel_data_handlers.util import apply_voi_lut

# Force GDCM handler for JPEG2000
pydicom.config.pixel_data_handlers = ['gdcm']

# ===== PATHS =====
csv_path   = "/root/project/MedViLL/data/bone/physionet.org/files/vindr-spinexr/1.0.0/annotations/train.csv"
dicom_dir  = "/root/project/MedViLL/data/bone/physionet.org/files/vindr-spinexr/1.0.0/train_images"
output_dir = "/root/project/MedViLL/data/bone/train"

os.makedirs(output_dir, exist_ok=True)

# Load CSV to get all unique image_ids
df = pd.read_csv(csv_path)
image_ids = df["image_id"].unique()

print(f"Total images to convert: {len(image_ids)}")
print(f"Using {multiprocessing.cpu_count()} CPU cores (will use 8 workers)")

def convert_dicom_to_jpg(image_id):
    dicom_path = os.path.join(dicom_dir, f"{image_id}.dicom")
    save_path  = os.path.join(output_dir, f"{image_id}.jpg")

    # Skip if already converted
    if os.path.exists(save_path):
        return image_id, True

    try:
        # Read DICOM
        dcm = pydicom.dcmread(dicom_path, force=True)

        # Apply VOI LUT if available
        img = apply_voi_lut(dcm.pixel_array, dcm)

        # Rescale slope/intercept
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            img = img.astype(np.float32) * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
        else:
            img = img.astype(np.float32)

        # Invert if MONOCHROME1
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img

        # Crop black borders (assume pixels <= 5 are background)
        coords = np.column_stack(np.where(img > 5))
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            img = img[y0:y1+1, x0:x1+1]

        # Convert to 8-bit (keep relative contrast)
        img_min, img_max = img.min(), img.max()
        img_norm = ((img - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)

        # Resize to keep X-ray content (optional: set max dimension)
        max_dim = 1024  # you can change
        h, w = img_norm.shape
        scale = min(max_dim/h, max_dim/w, 1.0)
        new_size = (int(w*scale), int(h*scale))
        img_resized = np.array(Image.fromarray(img_norm).resize(new_size, Image.BILINEAR))

        # Save as JPG
        Image.fromarray(img_resized).save(save_path, quality=95)

        return image_id, True

    except Exception as e:
        return image_id, f"Error: {e}"


# ====================== MULTI-PROCESSING ======================
num_workers = 8  # adjust based on your CPU

success = 0
failed = 0

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    future_to_id = {executor.submit(convert_dicom_to_jpg, iid): iid for iid in image_ids}
    
    for future in tqdm(as_completed(future_to_id), total=len(image_ids), desc="Converting DICOM to JPG"):
        image_id, result = future.result()
        if result is True:
            success += 1
        else:
            failed += 1
            print(f"Failed {image_id}: {result}")

print("\n✅ Conversion finished!")
print(f"Success: {success}")
print(f"Failed : {failed}")
print(f"Output folder: {output_dir}")