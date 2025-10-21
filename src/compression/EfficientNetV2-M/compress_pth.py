import os
import sys
import torch
import torch.nn as nn
import argparse


def compress_state_dict(state_dict, quantize_bits=8):
    """
    Compress model weights using quantization
    
    Args:
        state_dict: Model state dictionary
        quantize_bits: Number of bits for quantization (8 or 16)
    
    Returns:
        Compressed state dictionary
    """
    compressed_dict = {}
    total_original_size = 0
    total_compressed_size = 0
    
    for key, tensor in state_dict.items():
        original_size = tensor.element_size() * tensor.nelement()
        total_original_size += original_size
        
        # Only quantize float tensors (weights and biases)
        if tensor.dtype == torch.float32 and tensor.numel() > 1:
            if quantize_bits == 8:
                # Quantize to int8
                min_val = tensor.min()
                max_val = tensor.max()
                scale = (max_val - min_val) / 255.0
                
                if scale > 0:
                    quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
                    compressed_dict[key] = {
                        'quantized': quantized,
                        'min': min_val,
                        'scale': scale,
                        'shape': tensor.shape,
                        'dtype': 'uint8'
                    }
                    compressed_size = quantized.element_size() * quantized.nelement()
                    compressed_size += 8 + 8  # min and scale as float32
                else:
                    # If scale is 0, keep original (constant tensor)
                    compressed_dict[key] = tensor
                    compressed_size = original_size
            elif quantize_bits == 16:
                # Convert to float16
                compressed_dict[key] = tensor.half()
                compressed_size = compressed_dict[key].element_size() * compressed_dict[key].nelement()
            else:
                compressed_dict[key] = tensor
                compressed_size = original_size
        else:
            # Keep non-float tensors as is
            compressed_dict[key] = tensor
            compressed_size = original_size
        
        total_compressed_size += compressed_size
    
    compression_ratio = (1 - total_compressed_size / total_original_size) * 100
    print(f"Compression: {total_original_size / (1024**2):.2f} MB -> {total_compressed_size / (1024**2):.2f} MB ({compression_ratio:.1f}% reduction)")
    
    return compressed_dict

def decompress_state_dict(compressed_dict):
    """Decompress the state dictionary back to float32"""
    state_dict = {}
    
    for key, value in compressed_dict.items():
        if isinstance(value, dict) and 'quantized' in value:
            # Dequantize
            quantized = value['quantized']
            min_val = value['min']
            scale = value['scale']
            
            decompressed = quantized.float() * scale + min_val
            state_dict[key] = decompressed.reshape(value['shape'])
        else:
            # Convert float16 back to float32 if needed
            if isinstance(value, torch.Tensor) and value.dtype == torch.float16:
                state_dict[key] = value.float()
            else:
                state_dict[key] = value
    
    return state_dict

def load_and_convert(checkpoint_path, output_path, quantize_bits=8, verify=True):
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return False

    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Model Compression Pipeline")
    print(f"{'='*60}")
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Checkpoint size: {checkpoint_size:.2f} MB")
    print(f"Quantization: {quantize_bits}-bit")

    try:
        # Load checkpoint
        print("\nLoading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
            print("✓ Found 'model' key in checkpoint")
        else:
            model_state_dict = checkpoint
            print("⚠ No 'model' key found, using entire checkpoint as state dict")

        # Calculate original size
        original_params = sum(p.numel() for p in model_state_dict.values() if isinstance(p, torch.Tensor))
        print(f"✓ Model parameters: {original_params:,}")
        
        # Compress the model
        print(f"\nApplying {quantize_bits}-bit quantization...")
        compressed_dict = compress_state_dict(model_state_dict, quantize_bits=quantize_bits)
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save compressed model
        print(f"\nSaving compressed model to: {output_path}")
        torch.save(compressed_dict, output_path)

        output_size = os.path.getsize(output_path) / (1024 * 1024)
        total_reduction = ((checkpoint_size - output_size) / checkpoint_size) * 100
        print(f"\n{'='*60}")
        print(f"✓ Compression Complete!")
        print(f"{'='*60}")
        print(f"Original: {checkpoint_size:.2f} MB")
        print(f"Compressed: {output_size:.2f} MB")
        print(f"Total reduction: {total_reduction:.1f}%")
        print(f"Quantization: {quantize_bits}-bit")

        # Verify compression
        if verify:
            print(f"\nVerifying compression...")
            try:
                loaded_compressed = torch.load(output_path, map_location="cpu")
                
                # Decompress for verification
                decompressed_dict = decompress_state_dict(loaded_compressed)
                
                if set(model_state_dict.keys()) == set(decompressed_dict.keys()):
                    print(f"✓ Verification passed: {len(decompressed_dict)} keys match")
                    
                    # Calculate approximation error
                    max_error = 0
                    for key in model_state_dict.keys():
                        if isinstance(model_state_dict[key], torch.Tensor) and isinstance(decompressed_dict[key], torch.Tensor):
                            if model_state_dict[key].shape == decompressed_dict[key].shape:
                                error = (model_state_dict[key].float() - decompressed_dict[key].float()).abs().max().item()
                                max_error = max(max_error, error)
                    
                    print(f"✓ Max quantization error: {max_error:.6f}")
                    
                    # Try loading into Detector model
                    try:
                        from model import Detector
                        print("✓ Found model.py, verifying architecture compatibility...")
                        
                        # Create model instance
                        model = Detector()
                        
                        # Try loading with strict=False to handle potential mismatches
                        missing_keys, unexpected_keys = model.load_state_dict(decompressed_dict, strict=False)
                        
                        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                            print("✓ Model architecture verification: Perfect match!")
                        else:
                            if missing_keys:
                                print(f"⚠ Missing keys in checkpoint: {len(missing_keys)}")
                                if len(missing_keys) <= 5:
                                    for key in missing_keys:
                                        print(f"  - {key}")
                            if unexpected_keys:
                                print(f"⚠ Unexpected keys in checkpoint: {len(unexpected_keys)}")
                                if len(unexpected_keys) <= 5:
                                    for key in unexpected_keys:
                                        print(f"  - {key}")
                            print("✓ Model loaded successfully with partial match (strict=False)")
                        
                        # Test forward pass to ensure model works
                        model.eval()
                        with torch.no_grad():
                            test_input = torch.randn(1, 3, 384, 384)
                            output = model(test_input)
                            print(f"✓ Forward pass test successful: output shape {output.shape}")
                            
                    except ImportError:
                        print("⚠ Model verification skipped: model.py not found in current directory")
                    except Exception as e:
                        print(f"⚠ Model architecture verification failed: {str(e)}")
                        print("   Note: This may be expected if checkpoint was trained with different config")
                else:
                    print("⚠ Warning: Key mismatch detected between original and decompressed")
            except Exception as e:
                print(f"⚠ Verification failed: {e}")
        
        print(f"{'='*60}\n")
        return True

    except Exception as e:
        print(f"\n✗ Error during compression: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(args):
    if not args.input_checkpoint or not args.output_model:
        print("Error: Input and output paths are required.")
        return 1

    if not args.output_model.endswith(".pth"):
        args.output_model += ".pth"

    success = load_and_convert(
        checkpoint_path=args.input_checkpoint,
        output_path=args.output_model,
        quantize_bits=args.quantize_bits,
        verify=not args.no_verify,
    )

    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress .tar checkpoint files using quantization and save as .pth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 8-bit quantization (best compression, ~75% size reduction)
  python compress_pth.py checkpoint.tar compressed_model.pth
  
  # 16-bit quantization (float16, ~50% size reduction)
  python compress_pth.py checkpoint.tar model.pth --quantize-bits 16
  
  # No quantization (just extract weights, minimal reduction)
  python compress_pth.py checkpoint.tar model.pth --quantize-bits 32
  
  # Skip verification (faster)
  python compress_pth.py checkpoint.tar model.pth --no-verify
  
  # Compress from different directory
  python compress_pth.py ../../output/run_name/weights/best.tar weights/compressed.pth
        """,
    )

    parser.add_argument("input_checkpoint", help="Path to input .tar checkpoint file")
    parser.add_argument("output_model", help="Path to output .pth model file")
    parser.add_argument(
        "--quantize-bits",
        type=int,
        choices=[8, 16, 32],
        default=8,
        help="Quantization bits: 8 (best compression), 16 (float16), 32 (no quantization, default: 8)"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification after conversion"
    )

    args = parser.parse_args()
    sys.exit(main(args))