#!/usr/bin/env python3
"""
Script to export a trained SCNN model to ONNX format for TensorRT inference.

Uses fixed input size. Default 288x952 preserves KITTI aspect ratio
(370x1226 -> 288x952, divisible by 8).

SCNN produces two outputs:
  - seg_pred: Segmentation logits (1, 5, H, W)
  - exist_pred: Lane existence logits (1, 4)

Usage:
    python export_scnn_to_onnx.py --checkpoint /path/to/best.pth --output-dir onnxs
"""

import argparse
import os

from scnn_torch.model import SCNN

import torch
from torch.nn import Module


class SCNNWrapper(Module):
    """Wrapper for SCNN model for ONNX export."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        print('Loading SCNN model from checkpoint...')
        self.model = SCNN(ms_ks=9, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

        print(f'Loaded from iteration {checkpoint.get("iteration", "unknown")}')

    def forward(self, x):
        seg_pred, exist_pred = self.model(x)
        return seg_pred, exist_pred


def export_scnn_model(checkpoint_path: str, output_path: str, input_height: int, input_width: int):
    """Export SCNN model to ONNX format."""
    print('Creating SCNN model wrapper...')
    model = SCNNWrapper(checkpoint_path)

    print('Preparing dummy input...')
    dummy_input = torch.randn(1, 3, input_height, input_width)

    print('Exporting to ONNX...')
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            input_names=['input'],
            output_names=['seg_pred', 'exist_pred'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'seg_pred': {0: 'batch_size'},
                'exist_pred': {0: 'batch_size'}
            },
            dynamo=False,
            verbose=True
        )
        print(f'ONNX model saved to: {output_path}')
    except Exception as e:
        print(f'ONNX export failed: {e}')
        raise

    # Test the exported model
    print('\nTesting ONNX model...')
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('ONNX model validation passed')
    except ImportError:
        print('ONNX package not available - skipping model validation')
        print('Install with: pip install onnx')
    except Exception as e:
        print(f'ONNX model validation failed: {e}')


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--height', type=int, default=288,
                    help='The height of the input image (default: 288)')
    ap.add_argument('--width', type=int, default=952,
                    help='The width of the input image (default: 952 for KITTI aspect ratio)')
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='Path to trained SCNN checkpoint')
    ap.add_argument('--output-dir', type=str, default='onnxs',
                    help='The path to output ONNX file')
    args = vars(ap.parse_args())

    # Create output directory if it doesn't exist
    os.makedirs(args['output_dir'], exist_ok=True)

    height = args['height']
    width = args['width']
    checkpoint = args['checkpoint']
    output_dir = args['output_dir']

    # Export to ONNX
    print(f'=== Exporting SCNN for input size: {height}x{width} ===')
    output_path = os.path.join(output_dir, f'scnn_vgg16_{height}x{width}.onnx')
    export_scnn_model(
        checkpoint_path=checkpoint,
        output_path=output_path,
        input_height=height,
        input_width=width
    )

    print('ONNX export completed.')
