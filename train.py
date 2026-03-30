"""HW1 Training Script: Image Classification using ResNet-101.

Features:
- ResNet-101 Backbone (ImageNet1K_V2 weights)
- Custom Classification Head: Linear -> BatchNorm1d -> GELU -> Dropout -> Linear
- Full Fine-tuning from Epoch 1
- AdamW Optimizer + CosineAnnealingLR
- MixUp & CutMix + RandAugment + Label Smoothing
"""

import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data as torch_data
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


# Constants
BATCH_SIZE = 64
EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 0.05
DATA_DIR = './data'
NUM_CLASSES = 100
SAVE_PATH = 'best_resnet101_full_ft.pth'


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Performs MixUp data augmentation.

    Args:
        x: The input image batch tensor.
        y: The target label batch tensor.
        alpha: The alpha parameter for the Beta distribution.

    Returns:
        A tuple containing:
            - mixed_x: The mixed input tensor.
            - y_a: The original target label tensor.
            - y_b: The shuffled target label tensor.
            - lam: The mixing ratio lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, float(lam)


def rand_bbox(
    size: torch.Size, lam: float
) -> Tuple[int, int, int, int]:
    """Generates a random bounding box for CutMix.

    Args:
        size: The size of the input tensor.
        lam: The mixing ratio lambda.

    Returns:
        A tuple of (bbx1, bby1, bbx2, bby2) coordinates.
    """
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Performs CutMix data augmentation.

    Args:
        x: The input image batch tensor.
        y: The target label batch tensor.
        alpha: The alpha parameter for the Beta distribution.

    Returns:
        A tuple containing:
            - x: The augmented input tensor.
            - y_a: The original target label tensor.
            - y_b: The shuffled target label tensor.
            - lam: The effective mixing ratio lambda.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

    return x, y_a, y_b, float(lam)


def main() -> None:
    """Executes the main training pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'),
        train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'),
        val_transforms
    )

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print('Loading ResNet-101 (ImageNet_V2) for full fine-tuning...')
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)

    in_features = model.fc.in_features
    hidden_dim = 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(hidden_dim, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch + 1}/{EPOCHS} [Train]'
        )
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            r_val = np.random.rand()
            if r_val < 0.33:
                inputs, targets_a, targets_b, lam = cutmix_data(
                    inputs, labels, alpha=1.0
                )
                outputs = model(inputs)
                loss = (
                    criterion(outputs, targets_a) * lam
                    + criterion(outputs, targets_b) * (1. - lam)
                )
                _, predicted = outputs.max(1)

                correct_a = predicted.eq(targets_a).sum().item()
                correct_b = predicted.eq(targets_b).sum().item()
                train_correct += (lam * correct_a + (1 - lam) * correct_b)

            elif r_val < 0.66:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, labels, alpha=1.0
                )
                outputs = model(inputs)
                loss = (
                    criterion(outputs, targets_a) * lam
                    + criterion(outputs, targets_b) * (1. - lam)
                )
                _, predicted = outputs.max(1)

                correct_a = predicted.eq(targets_a).sum().item()
                correct_b = predicted.eq(targets_b).sum().item()
                train_correct += (lam * correct_a + (1 - lam) * correct_b)

            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            train_pbar.set_postfix(
                loss=loss.item(),
                acc=(train_correct / train_total)
            )

        epoch_train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0.0
        val_total = 0
        val_pbar = tqdm(
            val_loader,
            desc=f'Epoch {epoch + 1}/{EPOCHS} [Val]'
        )

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        scheduler.step()

        print(f'\nEpoch {epoch + 1}/{EPOCHS} Summary:')
        print(
            f'Train Acc:      {epoch_train_acc:.4f} | '
            f'Loss: {avg_train_loss:.4f}'
        )
        print(
            f'Validation Acc: {epoch_val_acc:.4f} | '
            f'Loss: {avg_val_loss:.4f}'
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_checkpoint = {
                'model_state_dict': model.state_dict(),
                'classes': train_dataset.classes
            }
            torch.save(save_checkpoint, SAVE_PATH)
            print(f'🌟 Saved Best Model! (Val Acc: {best_val_acc:.4f})\n')


if __name__ == '__main__':
    main()
