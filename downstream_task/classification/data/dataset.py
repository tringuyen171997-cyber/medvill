# import json
# import numpy as np
# import os
# from PIL import Image

# import torch
# from torch.utils.data import Dataset

# from utils.utils import truncate_seq_pair, numpy_seed


# class JsonlDataset(Dataset):
#     def __init__(self, data_path, tokenizer, transforms, vocab, args):
#         self.data = [json.loads(l) for l in open(data_path)]
#         self.data_dir = os.path.dirname(data_path)
#         self.tokenizer = tokenizer
#         self.args = args
#         self.vocab = vocab
#         self.n_classes = len(args.labels)
#         self.text_start_token =  ["[SEP]"]

#         with numpy_seed(0):
#             for row in self.data:
#                 if np.random.random() < args.drop_img_percent:
#                     row["img"] = None

#         self.max_seq_len = args.max_seq_len
#         self.max_seq_len -= args.num_image_embeds

#         self.transforms = transforms

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sentence = (
#             self.text_start_token
#             + self.tokenizer(self.data[index]["text"])[
#                 : (self.max_seq_len - 1)
#             ] + self.text_start_token
#         )
#         segment = torch.zeros(len(sentence))
#         sentence = torch.LongTensor(
#             [
#                 self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
#                 for w in sentence
#             ]
#         )
#         if self.args.task_type == "multilabel":
#             label = torch.zeros(self.n_classes)
#             if self.data[index]["label"] == '':
#                 self.data[index]["label"] = "'Others'"
#             else:
#                 pass  
#             label[
#                 [self.args.labels.index(tgt) for tgt in self.data[index]["label"].split(', ')]
#             ] = 1
#         else:
#             pass

#         image = None
#         if self.data[index]["img"]:
#             image = Image.open(
#                 os.path.join(self.data_dir, self.data[index]["img"]))
#         else:
#             image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
#         image = self.transforms(image)

#         # The first SEP is part of Image Token.
#         segment = segment[1:]
#         sentence = sentence[1:]
#         # The first segment (0) is of images.
#         segment += 1

#         return sentence, segment, image, label

import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import numpy_seed


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1,
                saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, vocab, args, is_train=True):
        with open(data_path) as f:
            self.data = [json.loads(l) for l in f]

        self.data_dir  = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args      = args
        self.vocab     = vocab
        self.n_classes = len(args.labels)
        self.is_train  = is_train
        self.transforms = get_transforms(is_train)

        # Max text length after reserving space for image tokens
        self.max_seq_len = args.max_seq_len - args.num_image_embeds

        # Randomly drop images during training
        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        # Print class distribution
        from collections import Counter
        label_counts = Counter()
        for d in self.data:
            for lbl in d['label'].split(', '):
                label_counts[lbl.strip()] += 1
        print(f"\n[DATASET] {data_path} — {len(self.data)} samples")
        print(f"  Label distribution:")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<40}: {cnt}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        # ── Text ───────────────────────────────────────────────────
        # Tokenize and truncate
        tokens = self.tokenizer.tokenize(d["text"])
        tokens = tokens[: self.max_seq_len - 1]  # leave room for SEP
        tokens = ["[SEP]"] + tokens + ["[SEP]"]

        segment  = torch.zeros(len(tokens), dtype=torch.long)
        sentence = torch.LongTensor([
            self.vocab.stoi[w] if w in self.vocab.stoi
            else self.vocab.stoi["[UNK]"]
            for w in tokens
        ])

        # Remove leading SEP (it belongs to image token side)
        segment  = segment[1:]
        sentence = sentence[1:]
        segment  = segment + 1  # text segment = 1

        # Attention mask (1 for real tokens, 0 for pad)
        attn_mask = torch.ones(len(sentence), dtype=torch.long)

        # Pad to max_seq_len
        pad_len  = self.max_seq_len - len(sentence)
        if pad_len > 0:
            pad_ids  = torch.zeros(pad_len, dtype=torch.long)
            seg_pad  = torch.zeros(pad_len, dtype=torch.long)
            mask_pad = torch.zeros(pad_len, dtype=torch.long)
            sentence  = torch.cat([sentence, pad_ids])
            segment   = torch.cat([segment,  seg_pad])
            attn_mask = torch.cat([attn_mask, mask_pad])

        # ── Label ──────────────────────────────────────────────────
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            raw   = d.get("label", "")
            if raw == '' or raw is None:
                raw = "No finding"
            for tgt in raw.split(', '):
                tgt = tgt.strip()
                if tgt in self.args.labels:
                    label[self.args.labels.index(tgt)] = 1
        else:
            # Single label classification
            raw   = d.get("label", "No finding").strip()
            label = torch.tensor(
                self.args.labels.index(raw) if raw in self.args.labels else 0,
                dtype=torch.long
            )

        # ── Image ──────────────────────────────────────────────────
        img_path = d.get("img")
        # print(img_path)
        if img_path:
            full_path = img_path
            try:
                image = Image.open(full_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot open {full_path}: {e}")
                image = Image.fromarray(
                    np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            image = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8))

        image = self.transforms(image)

        return sentence, segment, attn_mask, image, label