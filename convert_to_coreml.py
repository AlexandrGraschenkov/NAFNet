#!/usr/bin/env python3
"""
Convert NAFNet model to CoreML format for iOS deployment.

This script loads a trained NAFNet model and converts it to CoreML format,
handling the specific requirements for image restoration tasks.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import coremltools as ct
import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.models import create_model
from basicsr.utils.options import parse as parse_options


def load_nafnet_model(opt_yaml: str, weights_path: str) -> torch.nn.Module:
    """Load NAFNet model from configuration and weights."""
    opt = parse_options(opt_yaml, is_train=False)
    
    # Override weights path
    if 'path' not in opt:
        opt['path'] = {}
    opt['path']['pretrain_network_g'] = weights_path
    
    # Force CPU mode for conversion
    opt['dist'] = False
    opt['num_gpu'] = 0
    
    model_wrapper = create_model(opt)
    # Extract the actual PyTorch model from the wrapper
    model = model_wrapper.net_g
    model.eval()
    
    return model


def patch_layernorm_for_coreml(model: torch.nn.Module) -> None:
    """Replace LayerNorm2d with GroupNorm for CoreML compatibility."""
    from basicsr.models.archs.arch_util import LayerNorm2d
    
    for name, module in model.named_modules():
        if isinstance(module, LayerNorm2d):
            # Get the number of features from the weight parameter
            num_features = module.weight.shape[0] if hasattr(module, 'weight') else 1
            
            # Create GroupNorm replacement (groups=1 is equivalent to LayerNorm)
            groupnorm = torch.nn.GroupNorm(
                num_groups=1, 
                num_channels=num_features,
                eps=module.eps if hasattr(module, 'eps') else 1e-6,
                affine=True
            )
            
            # Copy parameters if they exist
            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                    groupnorm.weight.copy_(module.weight.data)
                if hasattr(module, 'bias') and module.bias is not None:
                    groupnorm.bias.copy_(module.bias.data)
            
            # Replace the module
            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split('.')[-1], groupnorm)


def patch_layernorm_for_coreml_2(model: torch.nn.Module) -> None:
    """Replace LayerNorm2d with supported GroupNorm(num_groups=1) during conversion."""
    from basicsr.models.archs.arch_util import LayerNorm2d

    for name, module in model.named_modules():
        if isinstance(module, LayerNorm2d):
            num_features = module.weight.shape[0] if hasattr(module, 'weight') else 1
            eps = module.eps if hasattr(module, 'eps') else 1e-6

            replacement = torch.nn.GroupNorm(
                num_groups=1,
                num_channels=num_features,
                eps=eps,
                affine=True,
            )

            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                    replacement.weight.copy_(module.weight.data)
                if hasattr(module, 'bias') and module.bias is not None:
                    replacement.bias.copy_(module.bias.data)

            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split('.')[-1], replacement)


class NAFNetWrapper(torch.nn.Module):
    """Wrapper for NAFNet to handle input/output preprocessing exactly like demo_nafnet_cpu.py."""
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, H, W] with values in [0, 1] (already preprocessed by CoreML)
        # The model expects values in [0, 1] range
        output = self.model(x) * 255.0
        
        # Keep output in CHW format for CoreML compatibility
        # CoreML will handle the final conversion
        return output


import coremltools as ct
from coremltools.models.neural_network import flexible_shape_utils
import os

def make_dyn_size(name):
    spec = ct.utils.load_spec(name)

    out_prev_name = spec.description.output._values[0].name
    ct.utils.rename_feature(spec, out_prev_name, 'output')

    img_size_ranges = flexible_shape_utils.NeuralNetworkImageSizeRange()
    img_size_ranges.add_height_range((16, 8192))
    img_size_ranges.add_width_range((16, 8192))
    flexible_shape_utils.update_image_size_range(spec, feature_name='image', size_range=img_size_ranges)
    flexible_shape_utils.update_image_size_range(spec, feature_name='output', size_range=img_size_ranges)

    n1, n2 = os.path.splitext(name)
    out_name = n1 + "_dsize" + n2
    ct.utils.save_spec(spec, out_name, weights_dir=name)

def convert_to_coreml(
    model: torch.nn.Module, 
    output_path: str, 
    input_channels: int = 3
) -> None:
    """Convert PyTorch model to CoreML format."""
    
    input_height: int = 768
    input_width: int = 768

    # Patch LayerNorm for CoreML compatibility
    # patch_layernorm_for_coreml_2(model)
    
    # Wrap model for proper input/output handling
    wrapped_model = NAFNetWrapper(model)
    wrapped_model.eval()
    
    # Create dummy input in [0, 1] range (CHW format)
    dummy_input = torch.rand(1, input_channels, input_height, input_width)
    
    from basicsr.models.archs.NAFNet_arch import update_test_mode
    update_test_mode(True)
    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input)
        traced_model = torch.jit.freeze(traced_model)
    

    input_shape = ct.Shape(
        shape=(1, 3,
            ct.RangeDim(lower_bound=32, upper_bound=8192, default=256),
            ct.RangeDim(lower_bound=32, upper_bound=8192, default=256))
    )
    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=input_shape,#dummy_input.shape,# 
                scale=1.0/255.0,  # Convert from [0, 255] to [0, 1]
                color_layout="RGB",
            )
        ],
        outputs=[
            ct.ImageType(
                name="output",
                color_layout="RGB"
            )
        ],
        # convert_to='neuralnetwork',
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )

    # mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, 16, "linear")
    # mlmodel = ct.compression_utils.affine_quantize_weights(mlmodel, mode="linear", dtype=np.float32)
    
    # Create output directory
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    
    # Save the model
    print(f"Saving CoreML model to {output_path}")
    mlmodel.save(output_path)
    # make_dyn_size(output_path)
    
    print("Conversion completed successfully!")


def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='Convert NAFNet model to CoreML')
    parser.add_argument(
        '--opt', 
        type=str, 
        default='options/test/GoPro/NAFNet-width32.yml',
        help='Path to YAML options file'
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default='weights/NAFNet-GoPro-width32.pth',
        help='Path to .pth weights file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='nafnet_gopro_width32.mlpackage',
        help='Output path for CoreML model'
    )
    parser.add_argument(
        '--input_channels', 
        type=int, 
        default=3,
        help='Number of input channels'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.opt):
        print(f"Error: Options file not found: {args.opt}")
        sys.exit(1)
        
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    print(f"Loading model from {args.opt} and {args.weights}")
    model = load_nafnet_model(args.opt, args.weights)
    
    convert_to_coreml(
        model, 
        args.output,
        args.input_channels
    )


if __name__ == '__main__':
    main()
