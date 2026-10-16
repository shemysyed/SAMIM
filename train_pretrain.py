from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer_torchvision import (
    MaskedVisionTransformerTorchvision,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PretrainDataset(Dataset):
    """
    CSV should contain at least:
    - image_path

    Optional columns:
    - patient_id
    """
    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        if "image_path" not in self.data.columns:
            raise ValueError("CSV must contain an 'image_path' column.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        image_path = self.data.iloc[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class SAMIMPretrainModel(nn.Module):
    """
    Starter SAMIM-style pretraining model:
    - ViT encoder
    - masked patch reconstruction
    - structure-aware loss target
    - consistency between two masked views

    This is a practical upload-ready version, not a final exact paper-faithful implementation.
    """
    def __init__(
        self,
        vit: torchvision.models.VisionTransformer,
        mask_ratio: float = 0.75,
        alpha: float = 0.5,
        lambda_struct: float = 1.0,
        lambda_cons: float = 1.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.alpha = alpha
        self.lambda_struct = lambda_struct
        self.lambda_cons = lambda_cons

        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.hidden_dim = vit.hidden_dim

        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.decoder = nn.Linear(self.hidden_dim, self.patch_size * self.patch_size * 3)

    def compute_structure_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Simple Sobel-like gradient magnitude approximation using finite differences.
        images: [B, C, H, W]
        returns: [B, 1, H, W]
        """
        gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]

        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        grad = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6)

        grad_min = grad.amin(dim=(2, 3), keepdim=True)
        grad_max = grad.amax(dim=(2, 3), keepdim=True)
        grad = (grad - grad_min) / (grad_max - grad_min + 1e-6)
        return grad

    def patch_score_from_structure(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns structure score per patch: [B, num_patches]
        """
        grad = self.compute_structure_map(images)
        patches = utils.patchify(grad.repeat(1, 3, 1, 1), self.patch_size)  # [B, N, patch_dim]
        score = patches.mean(dim=-1)  # [B, N]
        return score

    def patch_score_from_attention(self, images: torch.Tensor) -> torch.Tensor:
        """
        Practical approximation:
        use encoder token activations as a proxy for attention-based patch importance.
        Returns: [B, num_patches]
        """
        with torch.no_grad():
            tokens = self.backbone.vit._process_input(images)  # [B, N, D]
            score = tokens.abs().mean(dim=-1)  # [B, N]
        return score

    def informed_token_mask(
        self,
        images: torch.Tensor,
        use_informed_masking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns idx_keep, idx_mask for patch tokens only.
        sequence_length includes class token, so patch count = seq_length - 1
        """
        batch_size = images.shape[0]
        num_patches = self.sequence_length - 1
        num_mask = int(self.mask_ratio * num_patches)
        num_keep = num_patches - num_mask

        if not use_informed_masking:
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=images.device,
            )
            return idx_keep, idx_mask

        struct_score = self.patch_score_from_structure(images)  # [B, N]
        attn_score = self.patch_score_from_attention(images)    # [B, N]
        combined = self.alpha * attn_score + (1.0 - self.alpha) * struct_score

        # Higher score => more likely to be masked
        _, sorted_idx = torch.sort(combined, dim=1, descending=True)
        idx_mask_patch = sorted_idx[:, :num_mask]
        idx_keep_patch = sorted_idx[:, num_mask:]

        # Add 1 because Lightly masking expects token positions including CLS token
        idx_mask = idx_mask_patch + 1
        idx_keep = idx_keep_patch + 1

        # prepend cls token to keep indices
        cls_idx = torch.zeros((batch_size, 1), dtype=idx_keep.dtype, device=idx_keep.device)
        idx_keep = torch.cat([cls_idx, idx_keep], dim=1)

        return idx_keep, idx_mask

    def forward_single_view(
        self,
        images: torch.Tensor,
        use_informed_masking: bool = True,
    ):
        batch_size = images.shape[0]
        _, idx_mask = self.informed_token_mask(images, use_informed_masking=use_informed_masking)

        x_encoded = self.backbone.encode(images=images, idx_mask=idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        pred_pixels = self.decoder(x_encoded_masked)
        target_patches = utils.patchify(images, self.patch_size)
        target_pixels = utils.get_at_index(target_patches, idx_mask - 1)

        return pred_pixels, target_pixels, x_encoded, idx_mask

    def forward(
        self,
        images_view1: torch.Tensor,
        images_view2: torch.Tensor,
        use_informed_masking: bool = True,
    ):
        pred1, target1, enc1, idx_mask1 = self.forward_single_view(
            images_view1, use_informed_masking=use_informed_masking
        )
        pred2, target2, enc2, idx_mask2 = self.forward_single_view(
            images_view2, use_informed_masking=use_informed_masking
        )

        return {
            "pred1": pred1,
            "target1": target1,
            "pred2": pred2,
            "target2": target2,
            "enc1": enc1,
            "enc2": enc2,
            "idx_mask1": idx_mask1,
            "idx_mask2": idx_mask2,
        }

    def compute_losses(self, batch_dict, images_view1, images_view2):
        rec_loss1 = F.mse_loss(batch_dict["pred1"], batch_dict["target1"])
        rec_loss2 = F.mse_loss(batch_dict["pred2"], batch_dict["target2"])
        rec_loss = 0.5 * (rec_loss1 + rec_loss2)

        # structure-aware loss on patchified gradient targets
        struct_map1 = self.compute_structure_map(images_view1).repeat(1, 3, 1, 1)
        struct_map2 = self.compute_structure_map(images_view2).repeat(1, 3, 1, 1)

        struct_patches1 = utils.patchify(struct_map1, self.patch_size)
        struct_patches2 = utils.patchify(struct_map2, self.patch_size)

        struct_target1 = utils.get_at_index(struct_patches1, batch_dict["idx_mask1"] - 1)
        struct_target2 = utils.get_at_index(struct_patches2, batch_dict["idx_mask2"] - 1)

        struct_loss1 = F.mse_loss(batch_dict["pred1"], struct_target1)
        struct_loss2 = F.mse_loss(batch_dict["pred2"], struct_target2)
        struct_loss = 0.5 * (struct_loss1 + struct_loss2)

        # consistency on mean-pooled patch tokens (excluding cls)
        z1 = batch_dict["enc1"][:, 1:, :].mean(dim=1)
        z2 = batch_dict["enc2"][:, 1:, :].mean(dim=1)
        cons_loss = F.mse_loss(z1, z2.detach())

        total_loss = rec_loss + self.lambda_struct * struct_loss + self.lambda_cons * cons_loss

        return total_loss, rec_loss, struct_loss, cons_loss


@dataclass
class Args:
    csv_file: str
    save_dir: str
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1.5e-4
    weight_decay: float = 0.01
    mask_ratio: float = 0.75
    alpha: float = 0.5
    lambda_struct: float = 1.0
    lambda_cons: float = 1.0
    warmup_epochs: int = 10
    num_workers: int = 4
    seed: int = 42
    image_size: int = 224


def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_struct", type=float, default=1.0)
    parser.add_argument("--lambda_cons", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    return Args(**vars(args))


def build_transforms(image_size: int):
    base = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return base


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, name="samim_pretrain_last.pth"):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, os.path.join(save_dir, name))


def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    transform = build_transforms(args.image_size)
    dataset = PretrainDataset(csv_file=args.csv_file, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vit = torchvision.models.vit_b_16(weights=None)
    model = SAMIMPretrainModel(
        vit=vit,
        mask_ratio=args.mask_ratio,
        alpha=args.alpha,
        lambda_struct=args.lambda_struct,
        lambda_cons=args.lambda_cons,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("Starting SAMIM pretraining...")
    history = {"loss": [], "rec_loss": [], "struct_loss": [], "cons_loss": []}

    for epoch in range(args.epochs):
        model.train()
        start = time.time()

        total_loss = 0.0
        total_rec = 0.0
        total_struct = 0.0
        total_cons = 0.0

        use_informed_masking = epoch >= args.warmup_epochs

        for images in dataloader:
            images = images.to(device, non_blocking=True)

            # two views from same image tensor via lightweight augmentation noise
            images_view1 = images
            images_view2 = images + 0.01 * torch.randn_like(images)
            images_view2 = torch.clamp(images_view2, -3.0, 3.0)

            optimizer.zero_grad()

            outputs = model(
                images_view1,
                images_view2,
                use_informed_masking=use_informed_masking,
            )
            loss, rec_loss, struct_loss, cons_loss = model.compute_losses(
                outputs, images_view1, images_view2
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_struct += struct_loss.item()
            total_cons += cons_loss.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_rec = total_rec / len(dataloader)
        avg_struct = total_struct / len(dataloader)
        avg_cons = total_cons / len(dataloader)

        history["loss"].append(avg_loss)
        history["rec_loss"].append(avg_rec)
        history["struct_loss"].append(avg_struct)
        history["cons_loss"].append(avg_cons)

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"loss={avg_loss:.4f} rec={avg_rec:.4f} "
            f"struct={avg_struct:.4f} cons={avg_cons:.4f} "
            f"time={elapsed:.1f}s informed_mask={use_informed_masking}"
        )

        save_checkpoint(model, optimizer, scheduler, epoch, args.save_dir)

    torch.save(history, os.path.join(args.save_dir, "samim_pretrain_history.pth"))
    print("Pretraining complete.")


if __name__ == "__main__":
    main()
