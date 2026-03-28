import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
# from pytorch_pretrained_bert import BertTokenizer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import JsonlDataset
from data.vocab import Vocab


def get_transforms(args):
    if args.openi:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row = label_row.split(', ')

            label_freqs.update(label_row)
    else:
        pass
    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    """Build vocab from BiomedBERT tokenizer."""
    # ✅ Use HuggingFace AutoTokenizer instead of pytorch_pretrained_bert
    bert_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    vocab = Vocab()
    vocab.stoi    = bert_tokenizer.vocab
    vocab.itos    = {v: k for k, v in bert_tokenizer.vocab.items()}
    vocab.vocab_sz = len(vocab.stoi)
    return vocab

def collate_fn(batch, args):
    """Corrected collate for return order: (sentence, segment, attn_mask, image, label)"""
    texts = []
    segments = []
    masks = []
    images = []
    labels = []

    for row in batch:
        text, segment, mask, image, label = row
        texts.append(text)
        segments.append(segment)
        masks.append(mask)
        images.append(image)
        labels.append(label)

    # Stack everything
    text_tensor = torch.stack(texts)
    segment_tensor = torch.stack(segments)
    mask_tensor = torch.stack(masks)
    img_tensor = torch.stack(images)        # Now correctly gets the image tensor
    tgt_tensor = torch.stack(labels)

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor
# def collate_fn(batch, args):
#     lens = [len(row[0]) for row in batch]
#     bsz, max_seq_len = len(batch), max(lens)

#     mask_tensor = torch.zeros(bsz, max_seq_len).long()
#     text_tensor = torch.zeros(bsz, max_seq_len).long()
#     segment_tensor = torch.zeros(bsz, max_seq_len).long()

#     img_tensor = None
#     img_tensor = torch.stack([row[2] for row in batch])

#     if args.task_type == "multilabel":
#         # Multilabel case
#         tgt_tensor = torch.stack([row[3] for row in batch])
#     else:
#         # Single Label case
#         tgt_tensor = torch.cat([row[3] for row in batch]).long()

#     for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
#         tokens, segment = input_row[:2]
#         text_tensor[i_batch, :length] = tokens
#         segment_tensor[i_batch, :length] = segment
#         mask_tensor[i_batch, :length] = 1

#     return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    )
    # tokenizer = (
    #     BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize)
    # tokenizer = hf_tokenizer.tokenize
    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.Train_dset_name)
    )

    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, args.Train_dset_name),
        tokenizer,
        vocab,
        args,
    )

    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.Valid_dset_name),
        tokenizer,
        vocab,
        args,
    )
    def get_sample_weight(idx):
        d = train.data[idx]
        label_str = d.get("label", "No finding")
        labels_in_sample = [lbl.strip() for lbl in label_str.split(', ') if lbl.strip()]
        
        if not labels_in_sample:
            return 1.0
        
        # Use inverse frequency of the **rarest** label in this sample
        min_freq = min(args.label_freqs.get(l, 1) for l in labels_in_sample)
        return 1.0 / min_freq
    collate = functools.partial(collate_fn, args=args)
    # weights = [get_sample_weight(i) for i in range(len(train_dataset))]
    weights = [get_sample_weight(i) for i in range(len(train))]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    # train_loader = DataLoader(
    #     train,
    #     batch_size=args.batch_sz,
    #     shuffle=True,
    #     num_workers=args.n_workers,
    #     collate_fn=collate,
    # )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_sz,
        sampler=sampler,           
        num_workers=args.n_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dev,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader  # , test
