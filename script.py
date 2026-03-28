# import os
# import json
# from tqdm import tqdm

# # Paths
# jsonl_path = "/root/project/MedViLL/data/bone/Test.jsonl"
# train_dir = "/root/project/MedViLL/data/preprocessed/bone/test"

# # Read all IDs from Train.jsonl
# valid_ids = set()
# with open(jsonl_path, "r", encoding="utf-8") as f:
#     for line in f:
#         entry = json.loads(line)
#         valid_ids.add(entry["id"])

# # List all files in train_dir
# all_files = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]

# # Delete files not in Train.jsonl
# for filename in tqdm(all_files, desc="Cleaning train folder"):
#     image_id = filename.replace(".jpg", "")
#     if image_id not in valid_ids:
#         file_path = os.path.join(train_dir, filename)
#         os.remove(file_path)

# print("✅ Done! Only images listed in Train.jsonl remain in the train folder.")

import os
import json
from tqdm import tqdm

# Paths
jsonl_path = "/root/project/MedViLL/data/bone/Test.jsonl"
train_dir  = "/root/project/MedViLL/data/preprocessed/bone/test"

missing = []
total = 0

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Checking files"):
        entry = json.loads(line)
        image_id = entry["id"]
        filename = f"{image_id}.jpg"
        file_path = os.path.join(train_dir, filename)

        total += 1

        if not os.path.exists(file_path):
            missing.append(filename)

# Results
print(f"\nTotal entries in JSONL: {total}")
print(f"Missing files: {len(missing)}")

# Print some examples
if missing:
    print("\nFirst 20 missing files:")
    for f in missing[:20]:
        print(f)