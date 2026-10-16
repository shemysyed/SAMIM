from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet50,
    densenet121,
    ResNet50_Weights,
    DenseNet121_Weights,
)
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FinetuneDataset(Dataset):
    """
    CSV should contain:
    - image_path
    - label
    - fold

    Optional:
    - patient_id
    """
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SAMIMClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_ckpt: str):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("backbone.vit."):
                cleaned[k.replace("backbone.vit.", "")] = v
            elif k.startswith("backbone."):
                cleaned[k.replace("backbone.", "")] = v

        self.backbone.load_state_dict(cleaned, strict=False)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out


def build_model(model_name: str, num_classes: int, samim_ckpt: str = "") -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if model_name == "vit_b16":
        return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

    if model_name == "mae_vit_b16":
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        return model

    if model_name == "samim":
        if not samim_ckpt:
            raise ValueError("SAMIM finetuning requires --samim_ckpt")
        return SAMIMClassifier(num_classes=num_classes, pretrained_ckpt=samim_ckpt)

    raise ValueError(f"Unsupported model: {model_name}")


def load_mae_checkpoint(model: nn.Module, mae_ckpt: str):
    ckpt = torch.load(mae_ckpt, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def build_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def compute_multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):
    per_class_auc = {}
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        try:
            auc = roc_auc_score(y_true_bin, y_prob[:, c])
        except ValueError:
            auc = np.nan
        per_class_auc[c] = auc

    valid = [v for v in per_class_auc.values() if not np.isnan(v)]
    mauc = float(np.mean(valid)) if len(valid) > 0 else np.nan
    return per_class_auc, mauc


def evaluate_model(model, loader, device, num_classes):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)

    per_class_auc, mauc = compute_multiclass_auc(y_true, y_prob, num_classes)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    return {
        "acc": acc,
        "f1": f1,
        "recall": recall,
        "mauc": mauc,
        "per_class_auc": per_class_auc,
    }


@dataclass
class Args:
    csv_file: str
    save_dir: str
    model: str
    num_classes: int
    fold: int
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.05
    image_size: int = 224
    num_workers: int = 4
    seed: int = 42
    mae_ckpt: str = ""
    samim_ckpt: str = ""


def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet50", "densenet121", "vit_b16", "mae_vit_b16", "samim"])
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mae_ckpt", type=str, default="")
    parser.add_argument("--samim_ckpt", type=str, default="")
    args = parser.parse_args()
    return Args(**vars(args))


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv_file)
    required_cols = {"image_path", "label", "fold"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    train_df = df[df["fold"] != args.fold].copy()
    test_df = df[df["fold"] == args.fold].copy()

    # small validation split from train
    val_df = train_df.sample(frac=0.1, random_state=args.seed)
    train_df = train_df.drop(val_df.index)

    train_dataset = FinetuneDataset(train_df, transform=build_transforms(args.image_size, train=True))
    val_dataset = FinetuneDataset(val_df, transform=build_transforms(args.image_size, train=False))
    test_dataset = FinetuneDataset(test_df, transform=build_transforms(args.image_size, train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        samim_ckpt=args.samim_ckpt,
    )

    if args.model == "mae_vit_b16":
        if not args.mae_ckpt:
            raise ValueError("MAE finetuning requires --mae_ckpt")
        model = load_mae_checkpoint(model, args.mae_ckpt)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mauc = -1.0
    best_path = os.path.join(args.save_dir, f"{args.model}_fold{args.fold}_best.pth")

    print(f"Starting finetuning: model={args.model}, fold={args.fold}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device, args.num_classes)
        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start

        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"loss={avg_loss:.4f} val_mAUC={val_metrics['mauc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} time={elapsed:.1f}s"
        )

        if val_metrics["mauc"] > best_val_mauc:
            best_val_mauc = val_metrics["mauc"]
            torch.save({"model": model.state_dict()}, best_path)

    # final test evaluation
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    test_metrics = evaluate_model(model, test_loader, device, args.num_classes)

    out_path = os.path.join(args.save_dir, f"{args.model}_fold{args.fold}_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"acc: {test_metrics['acc']:.6f}\n")
        f.write(f"f1: {test_metrics['f1']:.6f}\n")
        f.write(f"recall: {test_metrics['recall']:.6f}\n")
        f.write(f"mauc: {test_metrics['mauc']:.6f}\n")
        for k, v in test_metrics["per_class_auc"].items():
            f.write(f"class_{k}_auc: {v:.6f}\n")

    print("Finetuning complete.")
    print(test_metrics)


if __name__ == "__main__":
    main()
