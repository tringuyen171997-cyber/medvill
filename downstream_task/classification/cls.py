import os
import csv
import argparse
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *


def get_args(parser):
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_sz", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    
    now = datetime.now().strftime('%Y-%m-%d')
    output_path = f"output/{now}"
    os.makedirs(output_path, exist_ok=True)
    parser.add_argument("--savedir", type=str, default=output_path)
    parser.add_argument("--save_name", type=str, default="spine_cls")
    parser.add_argument("--loaddir", type=str, default="output/YOUR_PRETRAIN_CHECKPOINT")

    # Dataset
    parser.add_argument("--data_path", type=str, default="data/bone")
    parser.add_argument("--Train_dset_name", type=str, default="Train.jsonl")
    parser.add_argument("--Valid_dset_name", type=str, default="Valid.jsonl")
    parser.add_argument("--Test_dset_name", type=str, default="Test.jsonl")

    # Model
    parser.add_argument("--embed_sz", type=int, default=768)
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--init_model", type=str, default="bert-base-scratch")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_image_embeds", type=int, default=64)
    parser.add_argument("--img_hidden_sz", type=int, default=768)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--weight_classes", type=int, default=1)

    # Freeze flags
    parser.add_argument("--freeze_img_all", type=bool, default=True)
    parser.add_argument("--freeze_txt_all", type=bool, default=True)
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--labels", nargs="+", type=str, required=True)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--openi", type=bool, default=False)


def plot_training_curves(args, train_losses, val_metrics):
    """Plot training curves and save to /root/project/MedViLL/output/plot/"""
    plot_dir = "/root/project/MedViLL/output/plot"
    os.makedirs(plot_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Train vs Val Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
    val_losses = [m.get("loss", 0) for m in val_metrics]
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Micro and Macro F1
    plt.subplot(2, 2, 2)
    micro_f1 = [m.get("micro_f1", 0) for m in val_metrics]
    macro_f1 = [m.get("macro_f1", 0) for m in val_metrics]
    plt.plot(epochs, micro_f1, 'g-', label='Micro F1', marker='o')
    plt.plot(epochs, macro_f1, 'orange', label='Macro F1', marker='s')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Micro and Macro ROC-AUC
    plt.subplot(2, 2, 3)
    micro_auc = [m.get("micro_roc_auc", 0) for m in val_metrics]
    macro_auc = [m.get("macro_roc_auc", 0) for m in val_metrics]
    plt.plot(epochs, micro_auc, 'purple', label='Micro AUC', marker='o')
    plt.plot(epochs, macro_auc, 'brown', label='Macro AUC', marker='s')
    plt.title('ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Per-class AUC (if available)
    plt.subplot(2, 2, 4)
    if val_metrics and 'classACC' in val_metrics[0]:
        # Just plot average of per-class AUCs for simplicity
        avg_class_auc = [np.mean(list(m.get('classACC', {}).values())) if m.get('classACC') else 0 
                        for m in val_metrics]
        plt.plot(epochs, avg_class_auc, 'c-', label='Avg Class AUC', marker='o')
        plt.title('Average Per-Class AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    base_name = f"{args.save_name}_training_curves"
    plt.savefig(os.path.join(plot_dir, f"{base_name}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, f"{base_name}.pdf"), bbox_inches='tight')
    
    print(f"[INFO] Training plots saved to: {plot_dir}/{base_name}.png")
    # plt.show()  # Uncomment if you want to display plots


# def get_criterion(args, device):
#     if args.task_type == "multilabel":
#         if args.weight_classes:
#             freqs = [args.label_freqs[l] for l in args.labels]
#             neg = [args.train_data_len - l for l in freqs]
#             weights = (torch.FloatTensor(freqs) / torch.FloatTensor(neg)) ** -1
#             return nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
#         return nn.BCEWithLogitsLoss()
#     return nn.CrossEntropyLoss(label_smoothing=0.1)

def get_criterion(args, device):
    if args.task_type == "multilabel":
        # Compute pos_weight per class:
        # pos_weight = num_negative / num_positive
        # Higher weight = model penalized more for missing rare class
        total = args.train_data_len
        pos_weights = []

        for lbl in args.labels:
            pos  = args.label_freqs.get(lbl, 1)
            neg  = total - pos
            # Cap weight at 20 to prevent extreme values for very rare classes
            w = min(neg / pos, 20.0)
            pos_weights.append(w)
            print(f"  pos_weight '{lbl}': {w:.2f}")

        pos_weight_tensor = torch.FloatTensor(pos_weights).to(device)
        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor,
            reduction='mean'
        )
    else:
        # Single label — use class weights
        total  = args.train_data_len
        weights = []
        for lbl in args.labels:
            freq = args.label_freqs.get(lbl, 1)
            weights.append(total / (len(args.labels) * freq))
        return nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weights).to(device),
            label_smoothing=0.1
        )


def get_optimizer(model, args):
    img_params = []
    bert_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'img_encoder' in name:
            img_params.append(p)
        elif any(x in name for x in ['txt_embeddings', 'encoder.layer', 'pooler']):
            bert_params.append(p)
        else:
            head_params.append(p)

    return AdamW([
        {'params': img_params, 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': bert_params, 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': head_params, 'lr': args.lr, 'weight_decay': 0.0},
    ], betas=(0.9, 0.999), eps=1e-6)


def model_forward(model, args, criterion, batch, device):
    txt, segment, mask, img, tgt = batch
    txt = txt.to(device)
    segment = segment.to(device)
    mask = mask.to(device)
    img = img.to(device)
    tgt = tgt.to(device)

    out = model(txt, mask, segment, img)
    loss = criterion(out, tgt)
    return loss, out, tgt


def model_eval(data, model, args, criterion, device, store_preds=False):
    model.eval()
    losses, preds, preds_bool, tgts = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating"):
            loss, out, tgt = model_forward(model, args, criterion, batch, device)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                prob = torch.sigmoid(out).cpu().numpy()
                pred_bool = prob > 0.5
                preds.append(prob)
                preds_bool.append(pred_bool)
            else:
                pred = out.argmax(dim=1).cpu().numpy()
                preds.append(pred)
            tgts.append(tgt.cpu().numpy())

    metrics = {"loss": np.mean(losses)}

    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        preds_bool = np.vstack(preds_bool)

        outAUROC = []
        for i in range(args.n_classes):
            try:
                outAUROC.append(roc_auc_score(tgts[:, i], preds[:, i]))
            except ValueError:
                outAUROC.append(0.0)

        classACC = {args.labels[i]: outAUROC[i] for i in range(args.n_classes)}
        metrics["micro_roc_auc"] = roc_auc_score(tgts, preds, average="micro")
        metrics["macro_roc_auc"] = roc_auc_score(tgts, preds, average="macro")
        metrics["macro_f1"] = f1_score(tgts, preds_bool, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds_bool, average="micro")
        metrics["classACC"] = classACC
        per_class_f1 = f1_score(tgts, preds_bool, average=None,
                                zero_division=0)
        print("\nPer-class AUC and F1:")
        for i, lbl in enumerate(args.labels):
            freq = args.label_freqs.get(lbl, 0)
            print(f"  {lbl:<30}: AUC={outAUROC[i]:.4f}  "
                  f"F1={per_class_f1[i]:.4f}  (n={freq})")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    return metrics, metrics.get("classACC", {}), tgts, preds


def train(args):
    print("Training start!")
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.save_name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)

    # Load pretrained weights if available
    pretrain_bin = os.path.join(args.loaddir, "pytorch_model.bin")
    if os.path.exists(pretrain_bin):
        pretrained = torch.load(pretrain_bin, map_location=device)
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        print(f"[INFO] Loaded pretrained weights from {args.loaddir}")
        print(f" Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        print("[WARN] No pretrained checkpoint found, training from scratch")

    criterion = get_criterion(args, device)
    optimizer = get_optimizer(model, args)

    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.max_epochs
    warmup_steps = int(total_steps * args.warmup)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger = create_logger(f"{args.savedir}/logfile.log", args)
    torch.save(args, os.path.join(args.savedir, "args.bin"))

    # History for plotting
    train_loss_history = []
    val_metrics_history = []

    best_metric = -np.inf
    n_no_improve = 0
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(args.max_epochs):
        model.train()
        train_losses = []
        optimizer.zero_grad()

        if epoch == 3:
            print("[INFO] Unfreezing encoders for full fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = get_optimizer(model, args)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train EP{epoch}")):
            loss, _, _ = model_forward(model, args, criterion, batch, device)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            train_losses.append(loss.item())
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Record train loss
        epoch_train_loss = np.mean(train_losses)
        train_loss_history.append(epoch_train_loss)

        # Evaluate
        metrics, classACC, tgts, preds = model_eval(val_loader, model, args, criterion, device)
        val_metrics_history.append(metrics)

        logger.info(f"EP{epoch} Train Loss: {epoch_train_loss:.4f}")
        log_metrics("Val", metrics, args, logger)

        tuning_metric = metrics["micro_f1"] if args.task_type == "multilabel" else metrics.get("acc", 0)
        
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
            }, True, args.savedir)
            print(f"[SAVED] Best model at EP{epoch} - metric={best_metric:.4f}")
        else:
            n_no_improve += 1

        if n_no_improve >= args.patience:
            print("Early stopping.")
            break

    # === Plot after training finishes ===
    print("\n" + "="*70)
    print("Training completed! Generating plots...")
    plot_training_curves(args, train_loss_history, val_metrics_history)
    print("="*70)


def cli_main():
    parser = argparse.ArgumentParser(description="Spine Classification")
    get_args(parser)
    args, remaining = parser.parse_known_args()
    args.n_classes = len(args.labels)
    assert remaining == [], remaining
    train(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()