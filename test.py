# # import torch
# # from PIL import Image
# # from transformers import AutoProcessor, LlavaForConditionalGeneration

# # # ===== CONFIG =====
# # model_id = "llava-hf/llava-1.5-7b-hf"
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # ===== LOAD MODEL =====
# # processor = AutoProcessor.from_pretrained(model_id)
# # model = LlavaForConditionalGeneration.from_pretrained(
# #     model_id,
# #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# #     low_cpu_mem_usage=True,
# # ).to(device)

# # model.eval()

# # # ===== LOAD IMAGE =====
# # image_path = "/root/project/MedViLL/data/bone/train/3de72e96a243b5de2a277fb50476fd31.jpg"
# # image = Image.open(image_path).convert("RGB")

# # # ===== LABEL & PROMPT (MUST include <image>) =====
# # label = "Osteophytes"

# # prompt = f"""USER: <image>
# # You are a professional radiologist. 
# # Given this spine X-ray and the known finding: "{label}", 
# # write a short, concise clinical report (1-2 sentences only). 
# # Be medically accurate and use standard radiology language.

# # ASSISTANT:"""

# # # ===== PREPARE INPUTS =====
# # inputs = processor(
# #     text=prompt,
# #     images=image,
# #     return_tensors="pt"
# # ).to(device)

# # # ===== GENERATE =====
# # with torch.no_grad():
# #     output = model.generate(
# #         **inputs,
# #         max_new_tokens=120,
# #         do_sample=True,
# #         temperature=0.7,
# #         top_p=0.9,
# #         repetition_penalty=1.1,
# #     )

# # # ===== DECODE =====
# # result = processor.decode(output[0], skip_special_tokens=True)

# # # Clean up the output (remove the prompt part)
# # generated_report = result.split("ASSISTANT:")[-1].strip()

# # print("\n=== GENERATED REPORT ===")
# # print(generated_report)
# import os
# import json
# import pandas as pd
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# # ========================= CONFIG =========================
# model_id = "llava-hf/llava-1.5-7b-hf"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# csv_path = "/root/project/MedViLL/data/bone/physionet.org/files/vindr-spinexr/1.0.0/annotations/test.csv"
# image_dir = "/root/project/MedViLL/data/preprocessed/bone/test"
# output_jsonl = "/root/project/MedViLL/data/bone/Test.jsonl"

# # ====================== SMARTER LABEL SELECTION ======================
# def select_best_label(lesion_list):
#     """
#     Rules (in priority order):
#     1. If only one label → use it (even if No finding)
#     2. If multiple → prefer any specific finding over 'No finding'
#     3. If multiple specific findings → join them (e.g. "Osteophytes, Fracture")
#     4. If ALL are 'No finding' → return 'No finding'
#     """
#     if not isinstance(lesion_list, list):
#         lesion_list = [lesion_list]

#     # Clean and deduplicate while preserving order
#     seen = set()
#     cleaned = []
#     for l in lesion_list:
#         l = str(l).strip()
#         if l not in seen:
#             seen.add(l)
#             cleaned.append(l)

#     # Separate findings from 'No finding'
#     specific = [l for l in cleaned if l.lower() != 'no finding']
#     no_finding = [l for l in cleaned if l.lower() == 'no finding']

#     if specific:
#         # Has real findings — join up to 3 to keep prompt concise
#         return ', '.join(specific[:3])
#     else:
#         return 'No finding'

# def build_label_map(df):
#     """Build image_id -> best_label mapping."""
#     label_map = {}
#     for image_id, group in df.groupby("image_id"):
#         lesion_list = group["lesion_type"].tolist()
#         label_map[image_id] = select_best_label(lesion_list)
#     return label_map

# # ====================== LOAD MODEL ======================
# print("Loading LLaVA model...")
# processor = AutoProcessor.from_pretrained(model_id)
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     low_cpu_mem_usage=True,
# ).to(device)
# model.eval()
# print("Model loaded!")

# # ====================== LOAD ANNOTATIONS ======================
# df = pd.read_csv(csv_path)
# label_map = build_label_map(df)

# # Quick sanity check on label distribution
# label_counts = pd.Series(label_map.values()).value_counts()
# print("\n[INFO] Label distribution (top 10):")
# print(label_counts.head(10))
# print(f"[INFO] Total unique images: {len(label_map)}\n")

# # ====================== GET ALL IMAGES ======================
# image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
# print(f"Found {len(image_files)} images to process.")

# # ====================== SMARTER PROMPT ======================
# def build_prompt(lesion):
#     if lesion.lower() == 'no finding':
#         return f"""USER: <image>
# You are a professional radiologist.
# This spine X-ray appears normal with no significant findings.
# Write a short clinical report (1-2 sentences) confirming normal appearance.
# Use standard radiology language.
# ASSISTANT:"""
#     else:
#         return f"""USER: <image>
# You are a professional radiologist.
# This spine X-ray shows evidence of: {lesion}.
# Write a short clinical report (1-2 sentences) describing these findings.
# Use standard radiology language and be medically accurate.
# ASSISTANT:"""

# # ====================== GENERATE JSONL ======================
# # Track stats
# stats = {'no_finding': 0, 'with_finding': 0, 'missing_label': 0}

# with open(output_jsonl, "w", encoding="utf-8") as f:
#     for img_name in tqdm(image_files, desc="Generating reports"):
#         image_id = img_name.replace(".jpg", "")

#         # Get best label
#         lesion = label_map.get(image_id, None)
#         if lesion is None:
#             stats['missing_label'] += 1
#             lesion = 'No finding'  # fallback
        
#         if lesion.lower() == 'no finding':
#             stats['no_finding'] += 1
#         else:
#             stats['with_finding'] += 1

#         try:
#             image = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
#         except Exception as e:
#             print(f"[WARN] Cannot open {img_name}: {e}, skipping.")
#             continue

#         prompt = build_prompt(lesion)
#         inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

#         with torch.no_grad():
#             output = model.generate(
#                 **inputs,
#                 max_new_tokens=120,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 repetition_penalty=1.1,
#             )

#         result = processor.decode(output[0], skip_special_tokens=True)
#         report = result.split("ASSISTANT:")[-1].strip()

#         entry = {
#             "id": image_id,
#             "split": "test",
#             "label": lesion,
#             "text": report,
#             "img": f"data/preprocessed/bone/test/{img_name}"
#         }
#         f.write(json.dumps(entry) + "\n")

#         # Clear GPU cache every 50 images
#         if stats['no_finding'] + stats['with_finding'] % 50 == 0:
#             torch.cuda.empty_cache()

# # ====================== FINAL STATS ======================
# print(f"\n✅ Done! JSONL saved to: {output_jsonl}")
# print(f"   Total processed  : {len(image_files)}")
# print(f"   With findings    : {stats['with_finding']}")
# print(f"   No finding       : {stats['no_finding']}")
# print(f"   Missing in CSV   : {stats['missing_label']}")



# import json
# import random
# import os
# from pathlib import Path

# # ======================
# # CONFIG
# # ======================
# train_jsonl_path = "/root/project/MedViLL/data/bone/Valid.jsonl"
# valid_jsonl_path = "/root/project/MedViLL/data/bone/Test.jsonl"

# # number of samples for validation
# num_valid = 300   # change to 30 if needed

# # OPTIONAL: image folders (set None if you don't want to move images)
# train_img_dir = "/root/project/MedViLL/data/preprocessed/bone/valid"
# valid_img_dir = "/root/project/MedViLL/data/preprocessed/bone/test"

# # ======================
# # LOAD DATA
# # ======================
# with open(train_jsonl_path, "r") as f:
#     data = [json.loads(line) for line in f]

# print(f"Total samples before split: {len(data)}")

# # ======================
# # SHUFFLE + SPLIT
# # ======================
# random.seed(42)
# random.shuffle(data)

# valid_data = data[:num_valid]
# train_data = data[num_valid:]

# print(f"Train samples after split: {len(train_data)}")
# print(f"Valid samples: {len(valid_data)}")

# # ======================
# # SAVE JSONL
# # ======================
# with open(valid_jsonl_path, "w") as f:
#     for item in valid_data:
#         f.write(json.dumps(item) + "\n")

# with open(train_jsonl_path, "w") as f:
#     for item in train_data:
#         f.write(json.dumps(item) + "\n")

# print("Saved Train.jsonl and Valid.jsonl")

# # ======================
# # OPTIONAL: MOVE IMAGES
# # ======================
# if train_img_dir and valid_img_dir:
#     Path(valid_img_dir).mkdir(parents=True, exist_ok=True)

#     moved = 0
#     for item in valid_data:
#         # ⚠️ Adjust this key depending on your JSON structure
#         # Common keys: "image", "image_path", "img"
#         img_name = item.get("image") or item.get("image_path") or item.get("img")

#         if img_name is None:
#             continue

#         src = os.path.join(train_img_dir, img_name)
#         dst = os.path.join(valid_img_dir, img_name)

#         if os.path.exists(src):
#             os.rename(src, dst)
#             moved += 1

#     print(f"Moved {moved} images to validation folder")


import json
import random
import os
from pathlib import Path
from collections import Counter

# ======================
# CONFIG
# ======================
source_jsonl_path = "/root/project/MedViLL/data/bone/Test.jsonl"   # source (2000 samples)
valid_jsonl_path  = "/root/project/MedViLL/data/bone/Valid.jsonl"  # output valid
test_jsonl_path   = "/root/project/MedViLL/data/bone/Test.jsonl"   # output test (overwrite)

num_valid = 500  # total valid samples you want
num_per_class = num_valid // 2  # 250 no_finding + 250 with_finding = balanced

source_img_dir = "/root/project/MedViLL/data/preprocessed/bone/test"
valid_img_dir  = "/root/project/MedViLL/data/preprocessed/bone/valid"

# ======================
# LOAD DATA
# ======================
with open(source_jsonl_path, "r") as f:
    data = [json.loads(line) for line in f]

print(f"Total samples: {len(data)}")

# ======================
# STRATIFIED SPLIT
# ======================
random.seed(42)

no_finding     = [d for d in data if str(d.get("label", "")).strip().lower() == "no finding"]
with_finding   = [d for d in data if str(d.get("label", "")).strip().lower() != "no finding"]

print(f"  No finding   : {len(no_finding)}")
print(f"  With finding : {len(with_finding)}")

# Shuffle both groups
random.shuffle(no_finding)
random.shuffle(with_finding)

# Check we have enough samples
if len(no_finding) < num_per_class:
    print(f"[WARN] Not enough no_finding samples ({len(no_finding)} < {num_per_class}), using all.")
    num_per_class = len(no_finding)

if len(with_finding) < num_per_class:
    print(f"[WARN] Not enough with_finding samples ({len(with_finding)} < {num_per_class}), using all.")
    num_per_class = min(num_per_class, len(with_finding))

# Take equal amounts from each class
valid_data = no_finding[:num_per_class] + with_finding[:num_per_class]
test_data  = no_finding[num_per_class:] + with_finding[num_per_class:]

# Shuffle final sets so they're not class-ordered
random.shuffle(valid_data)
random.shuffle(test_data)

print(f"\nAfter split:")
print(f"  Valid : {len(valid_data)} "
      f"(no_finding={sum(1 for d in valid_data if d['label'].lower() == 'no finding')}, "
      f"with_finding={sum(1 for d in valid_data if d['label'].lower() != 'no finding')})")
print(f"  Test  : {len(test_data)} "
      f"(no_finding={sum(1 for d in test_data if d['label'].lower() == 'no finding')}, "
      f"with_finding={sum(1 for d in test_data if d['label'].lower() != 'no finding')})")

# ======================
# SAVE JSONL
# ======================
with open(valid_jsonl_path, "w") as f:
    for item in valid_data:
        f.write(json.dumps(item) + "\n")

with open(test_jsonl_path, "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print(f"\nSaved Valid.jsonl ({len(valid_data)} samples)")
print(f"Saved Test.jsonl  ({len(test_data)} samples)")

# ======================
# MOVE IMAGES TO VALID FOLDER
# ======================
Path(valid_img_dir).mkdir(parents=True, exist_ok=True)

moved   = 0
missing = 0

for item in valid_data:
    img_name = item.get("img") or item.get("image") or item.get("image_path")
    if img_name is None:
        print(f"[WARN] No image key found in entry: {item.get('id')}")
        continue

    # Handle full path or just filename
    img_filename = os.path.basename(img_name)
    src = os.path.join(source_img_dir, img_filename)
    dst = os.path.join(valid_img_dir, img_filename)

    if os.path.exists(src):
        os.rename(src, dst)
        moved += 1
    elif os.path.exists(dst):
        pass  # already moved, skip silently
    else:
        missing += 1
        if missing <= 5:  # only print first 5 warnings
            print(f"[WARN] Image not found: {src}")

print(f"\nMoved  : {moved} images → {valid_img_dir}")
print(f"Missing: {missing} images not found")

# ======================
# FINAL LABEL DISTRIBUTION CHECK
# ======================
print("\n[INFO] Valid set label breakdown:")
valid_labels = Counter(d["label"] for d in valid_data)
for label, count in sorted(valid_labels.items(), key=lambda x: -x[1]):
    print(f"  {label:<40} : {count}")

print("\n[INFO] Test set label breakdown:")
test_labels = Counter(d["label"] for d in test_data)
for label, count in sorted(test_labels.items(), key=lambda x: -x[1]):
    print(f"  {label:<40} : {count}")

