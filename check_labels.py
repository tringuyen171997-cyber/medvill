# run_check_labels.py
import json
from collections import Counter

for split, path in [
    ("Train", "data/bone/Train.jsonl"),
    ("Valid", "data/bone/Valid.jsonl"),
    ("Test",  "data/bone/Test.jsonl"),
]:
    with open(path) as f:
        data = [json.loads(l) for l in f]
    
    label_counts = Counter()
    for d in data:
        for lbl in d['label'].split(', '):
            label_counts[lbl.strip()] += 1
    
    print(f"\n{split}:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  '{lbl}' : {cnt}")