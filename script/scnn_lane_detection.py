"""
Inference script for running SCNN on arbitrary images.

Usage:
    python scnn_lane_detection.py --checkpoint /path/to/best.pth --image test/image_000.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scnn_torch.model import SCNN
from scnn_torch.utils import resize_seg_pred, visualize_lanes

import torch
import torch.nn.functional as F
from torchvision import transforms


# ImageNet normalization constants
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description='SCNN Lane Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--target_height', type=int, default=288,
                        help='Target height for resizing (default: 288)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Lane existence threshold (default: 0.5)')
    return parser.parse_args()


def main():
    args = parse_args()

    # === Step 1: Load and preprocess image ===
    img_pil = Image.open(args.image).convert('RGB')
    original_size = (img_pil.height, img_pil.width)  # (H, W)

    # Calculate target width preserving aspect ratio, divisible by 8
    target_height = args.target_height
    target_width = round(original_size[1] * target_height / original_size[0] / 8) * 8

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # Apply transformations
    input_tensor = transform(img_pil).unsqueeze(0)  # Shape: [1, 3, H, W]

    # === Step 2: Load pretrained SCNN model ===
    model = SCNN(ms_ks=9, pretrained=False)

    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    print(f"Loaded from iteration {checkpoint.get('iteration', 'unknown')}")

    model.eval()

    # === Step 3: Run inference ===
    with torch.no_grad():
        seg_pred, exist_pred = model(input_tensor)

        # Convert logits to probabilities
        seg_pred = F.softmax(seg_pred, dim=1)
        exist_pred = torch.sigmoid(exist_pred)

    # Convert to numpy
    seg_pred = seg_pred.squeeze(0).cpu().numpy()  # Shape: [5, H, W]
    exist_pred = exist_pred.squeeze(0).cpu().numpy()  # Shape: [4]

    # === Step 4: Post-process ===
    # Resize seg_pred back to original image size
    seg_pred_resized = resize_seg_pred(seg_pred, original_size)

    # Print lane existence probabilities
    print(f"Lane existence probabilities: {[f'{p:.2f}' for p in exist_pred]}")

    # === Step 5: Visualize with matplotlib ===
    # Convert PIL image to numpy array for visualization
    img_np = np.array(img_pil)

    # Get overlay and lane mask using visualize_lanes
    img_overlay, lane_img = visualize_lanes(
        img_np, seg_pred_resized, exist_pred, threshold=args.threshold
    )

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(3, 1, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(img_overlay)
    plt.title('Overlay (Original + Lanes)')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(lane_img)
    plt.title('Lane Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
