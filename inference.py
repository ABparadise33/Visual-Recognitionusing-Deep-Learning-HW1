"""HW1 Inference Script: Image Classification using ResNet-101.

This script performs inference using 10-Crop Test Time Augmentation (TTA)
and generates a CSV file for CodaBench submission.
"""

import glob
import os
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils import data as torch_data
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


# Constants
TEST_BATCH_SIZE = 16
TEST_DIR = './data/test'
TRAIN_DIR = './data/train'
MODEL_WEIGHTS = 'best_resnet101_full_ft.pth'
NUM_CLASSES = 100
OUTPUT_FILE = 'prediction_resnet101_full_ft.csv'


class InferenceDataset(torch_data.Dataset):
    """Dataset class for loading test images for inference.

    Attributes:
        root_dir: The root directory containing the test images.
        transform: Optional transform to be applied on a sample.
        image_paths: A list of paths to the image files.
    """

    def __init__(
        self, root_dir: str, transform: Optional[transforms.Compose] = None
    ) -> None:
        """Initializes the InferenceDataset.

        Args:
            root_dir: Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        search_pattern = os.path.join(root_dir, '**', '*.*')
        all_paths = glob.glob(search_pattern, recursive=True)

        valid_exts = ('.png', '.jpg', '.jpeg')
        self.image_paths = [
            p for p in all_paths if p.lower().endswith(valid_exts)
        ]

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Fetches the sample at the given index.

        Args:
            idx: Index of the sample to fetch.

        Returns:
            A tuple containing the transformed image tensor and its name.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(img_path)
        image_name = os.path.splitext(filename)[0]

        return image, image_name


def main() -> None:
    """Executes the main inference pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('🚀 Starting 10-Crop TTA Inference (ResNet-101 Full Fine-tuning)...')

    try:
        dummy_dataset = datasets.ImageFolder(TRAIN_DIR)
        real_classes = dummy_dataset.classes
    except FileNotFoundError as e:
        print(f'Error reading {TRAIN_DIR}: {e}')
        return

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(
            lambda crops: torch.stack([
                normalize(transforms.ToTensor()(crop)) for crop in crops
            ])
        )
    ])

    test_dataset = InferenceDataset(TEST_DIR, transform=test_transforms)
    test_loader = torch_data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print('Initializing ResNet-101...')
    model = models.resnet101(weights=None)

    in_features = model.fc.in_features
    hidden_dim = 512

    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(hidden_dim, NUM_CLASSES)
    )

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Weights file '{MODEL_WEIGHTS}' not found!")
        return

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    image_names = []
    predictions = []
    test_pbar = tqdm(test_loader, desc='Predicting with 10-Crop TTA')

    with torch.no_grad():
        for inputs, names in test_pbar:
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).to(device)

            outputs = model(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

            _, predicted_indices = outputs_avg.max(1)
            predicted_indices = predicted_indices.cpu().numpy()

            image_names.extend(names)
            for idx in predicted_indices:
                predictions.append(int(real_classes[idx]))

    results_df = pd.DataFrame({
        'image_name': image_names,
        'pred_label': predictions
    })
    results_df.sort_values(by='image_name', inplace=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f'\n🎉 Successfully saved predictions to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
