import argparse
import json
import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import csv

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import prune
from tqdm import tqdm
import pywt

# Third-party deps
from retinaface.pre_trained_models import get_model

from preprocess import extract_frames
from datasets import (
    init_cdf, init_ff, init_dfd, init_dfdc, init_dfdcp, init_ffiw
)

warnings.filterwarnings('ignore')

# --- 1. MODEL & UTILS ---

def get_detector_class(backbone: str):
    try:
        if backbone == 'b4':
            from model import Detector4
            return Detector4
        elif backbone == 'b5':
            from model import Detector5
            return Detector5
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    except ImportError as e:
        print(f"Error importing model: {e}")
        sys.exit(1)

def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement()

def _state_dict_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += _tensor_nbytes(value)
    return total

# --- 2. COMPRESSION LOGIC ---

def is_quantized_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and {"quantized", "min", "scale", "shape"}.issubset(entry.keys())

def decompress_state_dict(compressed: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in compressed.items():
        if is_quantized_entry(value):
            quantized = value["quantized"].float()
            state_dict[key] = quantized * value["scale"] + value["min"]
            state_dict[key] = state_dict[key].reshape(value["shape"])
        elif isinstance(value, torch.Tensor) and value.dtype == torch.float16:
            state_dict[key] = value.float()
        else:
            state_dict[key] = value
    return state_dict

def quantize_state_dict(state_dict: Dict[str, torch.Tensor], quantize_bits: int) -> Tuple[Dict[str, Any], Dict[str, float]]:
    quantized: Dict[str, Any] = {}
    original_bytes = _state_dict_nbytes(state_dict)
    compressed_bytes = 0

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            quantized[key] = tensor
            continue

        if tensor.dtype != torch.float32 or tensor.numel() <= 1:
            quantized[key] = tensor
            compressed_bytes += _tensor_nbytes(tensor)
            continue

        if quantize_bits == 8:
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
            if scale <= 0:
                quantized[key] = tensor
                compressed_bytes += _tensor_nbytes(tensor)
                continue
            q_tensor = ((tensor - min_val) / scale).round().to(torch.uint8)
            quantized[key] = {"quantized": q_tensor, "min": min_val, "scale": scale, "shape": tensor.shape, "dtype": "uint8"}
            compressed_bytes += _tensor_nbytes(q_tensor) + 8 + 8
        elif quantize_bits == 16:
            half_tensor = tensor.half()
            quantized[key] = half_tensor
            compressed_bytes += _tensor_nbytes(half_tensor)
        else:
            quantized[key] = tensor
            compressed_bytes += _tensor_nbytes(tensor)

    compression_ratio = 1 - (compressed_bytes / max(original_bytes, 1))
    return quantized, {
        "original_mb": original_bytes / (1024 ** 2),
        "compressed_mb": compressed_bytes / (1024 ** 2),
        "compression_ratio": compression_ratio,
    }

def apply_global_pruning(model: nn.Module, amount: float) -> None:
    if amount <= 0: return
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    if not parameters_to_prune: return
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in parameters_to_prune:
        try:
            prune.remove(module, "weight")
        except ValueError: continue

# --- 3. IMAGING HELPERS (UPDATED) ---

def _face_chw_to_rgb(face: np.ndarray) -> np.ndarray:
    """Convert CHW face tensors into contiguous HWC RGB images."""
    face_arr = np.asarray(face)
    if face_arr.ndim == 3 and face_arr.shape[0] == 3:
        face_arr = np.transpose(face_arr, (1, 2, 0))
    return np.ascontiguousarray(face_arr)


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image data is uint8, handling both 0-1 and 0-255 floats."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    max_val = float(img.max()) if img.size else 0.0
    if max_val <= 1.0:
        img *= 255.0
    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8)


def _prepare_model_input(face_image: np.ndarray) -> np.ndarray:
    """Convert a face crop (CHW or HWC) into the detector's 3x380x380 input."""
    arr = np.asarray(face_image)
    if arr.ndim != 3:
        raise ValueError("Face image must be HWC or CHW")
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        face_rgb = np.transpose(arr, (1, 2, 0))
    else:
        face_rgb = arr
    face_norm = face_rgb.astype('float32') / 255.0
    b, g, r = cv2.split(face_norm)
    cA_r, _ = pywt.dwt2(r, 'sym2', mode='reflect')
    cA_g, _ = pywt.dwt2(g, 'sym2', mode='reflect')
    cA_b, _ = pywt.dwt2(b, 'sym2', mode='reflect')
    cA_r = cv2.resize(cA_r, (380, 380))
    cA_g = cv2.resize(cA_g, (380, 380))
    cA_b = cv2.resize(cA_b, (380, 380))
    r = cv2.resize(r, (380, 380))
    g = cv2.resize(g, (380, 380))
    b = cv2.resize(b, (380, 380))
    return np.array([(cA_r + r) / 2, (cA_g + g) / 2, (cA_b + b) / 2], dtype=np.float32)

class GradCAM:
    """
    Grad-CAM implementation for deepfake detection models.
    
    For facial deepfake detection, we generate CAM for the "fake" class (class 1)
    to visualize WHERE the model detected signs of manipulation.
    """
    def __init__(self, model: nn.Module, layer_name: str = 'net._conv_head'):
        self.model = model
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.layer = self._find_target_layer(layer_name)
        if self.layer is None:
            raise ValueError(f"Unable to locate target layer '{layer_name}' for GradCAM")
        self.activations = None
        self.gradients = None
        self._handles = []
        self._handles.append(self.layer.register_forward_hook(self._forward_hook))
        self._handles.append(self.layer.register_full_backward_hook(self._backward_hook))

    def _find_target_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Find target layer by name. Supports multiple naming conventions."""
        # For EfficientNet wrapped in Detector class: model.net._conv_head
        # Try multiple patterns to find the layer
        candidates = {
            layer_name,
            f"module.{layer_name}",
            layer_name.replace("net.", ""),  # In case model IS the net
        }
        for name, module in self.model.named_modules():
            if name in candidates:
                return module
        
        # Fallback: find _conv_head directly for EfficientNet models
        for name, module in self.model.named_modules():
            if name.endswith('_conv_head') and isinstance(module, nn.Conv2d):
                return module
        
        return None

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate(self, tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            tensor: Input image tensor (CHW or NCHW)
            target_class: Class to generate CAM for. Default is 1 (fake class)
                         For deepfake detection:
                         - class 0 = real
                         - class 1 = fake (we want to see WHERE fakeness was detected)
        
        Returns:
            Normalized heatmap as numpy array (0-1 range)
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        
        # Reset activations and gradients
        self.activations = None
        self.gradients = None
        
        with torch.enable_grad():
            tensor = tensor.clone().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            output = self.model(tensor)
            
            # Get score for the target class (fake class = 1 for deepfake detection)
            score = output[:, target_class].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            return np.zeros(tensor.shape[-2:], dtype=np.float32)

        # Global average pooling on gradients to get channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions
        
        # Upsample to input size
        cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalize to 0-1 range
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam

def _apply_gaussian_blur(heatmap: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """Apply Gaussian blur to smooth the heatmap for better visualization."""
    if blur_radius % 2 == 0:
        blur_radius += 1  # Must be odd
    return cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)


def save_heatmap_overlay(model, device, video_path, save_path, face_detector, 
                         face_rgb: Optional[np.ndarray] = None, 
                         gradcam: Optional[GradCAM] = None,
                         blur_radius: int = 15,
                         heatmap_intensity: float = 0.5,
                         force_regenerate: bool = False):
    """
    Generates a heatmap overlay showing WHERE the model detected deepfake artifacts.
    
    The heatmap highlights facial regions that contributed to the fake detection,
    such as blending boundaries, texture inconsistencies, or manipulation artifacts.
    
    Args:
        model: The deepfake detection model
        device: Torch device
        video_path: Path to video file
        save_path: Output path for the overlay image
        face_detector: RetinaFace detector
        face_rgb: Optional pre-extracted face image (RGB, HWC, uint8)
        gradcam: Optional pre-initialized GradCAM object
        blur_radius: Gaussian blur radius for smoothing heatmap (odd number)
        heatmap_intensity: Blending intensity (0-1), higher = more heatmap visible
        force_regenerate: If True, regenerate even if file exists
    """
    if save_path.exists() and not force_regenerate: 
        return True
    try:
        if face_rgb is None:
            faces, _ = extract_frames(video_path, 1, face_detector)
            if len(faces) == 0: 
                return False
            face_rgb = _ensure_uint8(_face_chw_to_rgb(faces[0]))

        model_input = _prepare_model_input(face_rgb)
        tensor = torch.from_numpy(model_input)
        
        cam_helper = gradcam or GradCAM(model)
        # Generate CAM for fake class (class 1) to see WHERE fakeness was detected
        heatmap = cam_helper.generate(tensor, target_class=1)

        # Resize heatmap to match face image dimensions
        heatmap = cv2.resize(heatmap, (face_rgb.shape[1], face_rgb.shape[0]))
        
        # Apply Gaussian blur to smooth the heatmap for better visualization
        heatmap = _apply_gaussian_blur(heatmap, blur_radius)
        
        # Re-normalize after blur
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max - heatmap_min > 1e-8:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        # Convert to colormap (JET: blue=low, red=high attention)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Convert face to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        
        # Blend: darken low-attention areas, highlight high-attention areas
        # This makes manipulation regions stand out more clearly
        alpha = heatmap_intensity
        superimposed = cv2.addWeighted(face_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)

        _ensure_dir(save_path.parent)
        cv2.imwrite(str(save_path), superimposed)
        return True
    except Exception as e:
        print(f"Heatmap generation failed: {e}")
        return False

def save_plain_face(video_path, save_path, face_detector, face_rgb: Optional[np.ndarray] = None):
    """Generates just the face crop (ensuring correct BGR/uint8)."""
    if save_path.exists(): return True 
    try:
        if face_rgb is None:
            faces, _ = extract_frames(video_path, 1, face_detector)
            if len(faces) == 0: return False
            face_rgb = _ensure_uint8(_face_chw_to_rgb(faces[0]))
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

        _ensure_dir(save_path.parent)
        cv2.imwrite(str(save_path), face_bgr)
        return True
    except Exception:
        return False

# --- 4. EVALUATION & ANALYTICS ENGINE ---

def load_predictions_from_csv(csv_path: Path) -> Optional[Dict[str, Any]]:
    if not csv_path.exists(): return None
    try:
        print(f"--> Found existing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        required_cols = {'sample_file', 'score', 'target'}
        if not required_cols.issubset(df.columns): return None
        predictions = df.to_dict('records')
        auc = roc_auc_score(df['target'], df['score'])
        print(f"--> Loaded {len(predictions)} predictions. Cached AUC: {auc:.4f}")
        return {"auc": float(auc), "predictions": predictions}
    except Exception: return None

def evaluate_model(model: nn.Module, dataset: str, device: torch.device, n_frames: int, max_samples: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if dataset == 'CDF': video_list, target_list = init_cdf()
    elif dataset == 'FF': video_list, target_list = init_ff("Face2Face")
    elif dataset == 'DFDC': video_list, target_list = init_dfdc()
    else: return None

    if not video_list: return None
    if max_samples:
        video_list = video_list[:max_samples]
        target_list = target_list[:max_samples]

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    try:
        gradcam = GradCAM(model)
    except ValueError as err:
        print(f"GradCAM init failed: {err}")
        gradcam = None
    
    output_scores = []
    predictions = [] 
    
    model.eval()
    for filename, target in tqdm(zip(video_list, target_list), desc=f"Evaluating", total=len(video_list)):
        try:
            face_list, _ = extract_frames(filename, n_frames, face_detector)
            if len(face_list) == 0:
                output_scores.append(0.5)
                predictions.append({"sample_file": filename, "score": 0.5, "target": target})
                continue

            gpu_batch = []
            with torch.no_grad():
                for f_img in face_list:
                    img_dwt = _prepare_model_input(f_img)
                    gpu_batch.append(torch.from_numpy(img_dwt).unsqueeze(0).to(device))
                
                if gpu_batch:
                    pred = model(torch.cat(gpu_batch, dim=0)).softmax(1)[:, 1]
                    final_score = float(pred.max().item()) 
                else: final_score = 0.5

            output_scores.append(final_score)
            predictions.append({"sample_file": filename, "score": final_score, "target": target})
            torch.cuda.empty_cache()
        except:
            output_scores.append(0.5)
            predictions.append({"sample_file": filename, "score": 0.5, "target": target})

    try:
        auc = roc_auc_score(target_list, output_scores)
        return {"auc": float(auc), "predictions": predictions}
    except: return None

def run_analytics_full(predictions: List[Dict], model, device, output_dir: Path, regenerate_heatmaps: bool = False) -> Dict:
    result_dir = output_dir / "result"
    
    # Bucket Directories - singular "image"
    cats = ["true_positives", "true_negatives", "false_positives", "false_negatives"]
    sub_types = ["image", "heatmap"]
    
    for cat in cats:
        for st in sub_types:
            _ensure_dir(result_dir / cat / st)
            
    # CSV (Root)
    _ensure_dir(result_dir)
    csv_path = result_dir / "predictions.csv" 
    if not csv_path.exists():
        df = pd.DataFrame(predictions)
        df.to_csv(csv_path, index=False)

    threshold = 0.5
    # Initialize rank data
    rank_data = {c: [] for c in cats}
    
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    
    try:
        gradcam = GradCAM(model)
    except ValueError as err:
        print(f"GradCAM init failed: {err}")
        gradcam = None

    print(f"Generating visuals for {len(predictions)} samples...")
    
    for i, item in tqdm(enumerate(predictions), total=len(predictions)):
        score = item['score']
        target = item['target']
        fname = Path(item['sample_file']).name
        save_name = f"{fname}.jpg"
        predicted_class = 1 if score >= threshold else 0
        
        category = ""
        if target == 1 and predicted_class == 1: category = "true_positives"
        elif target == 0 and predicted_class == 0: category = "true_negatives"
        elif target == 0 and predicted_class == 1: category = "false_positives"
        elif target == 1 and predicted_class == 0: category = "false_negatives"
        
        path_img = result_dir / category / "image" / save_name
        path_map = result_dir / category / "heatmap" / save_name
        
        faces_cached, _ = extract_frames(item['sample_file'], 1, face_detector)
        face_rgb = _ensure_uint8(_face_chw_to_rgb(faces_cached[0])) if faces_cached else None

        save_plain_face(item['sample_file'], path_img, face_detector, face_rgb=face_rgb)
        if gradcam is not None:
            save_heatmap_overlay(model, device, item['sample_file'], path_map, face_detector, face_rgb=face_rgb, gradcam=gradcam, force_regenerate=regenerate_heatmaps)
        
        entry = {
            "sample_id": i+1,
            "sample_file": item['sample_file'],
            "score": score,
            "target": target,
            "prediction": predicted_class,
            "image": f"{category}/image/{save_name}",
            "heatmap": f"{category}/heatmap/{save_name}"
        }
        
        rank_data[category].append(entry)

    # Sort
    rank_data["true_positives"].sort(key=lambda x: x['score'], reverse=True)
    rank_data["false_negatives"].sort(key=lambda x: x['score'], reverse=False)
    rank_data["false_positives"].sort(key=lambda x: x['score'], reverse=True)
    rank_data["true_negatives"].sort(key=lambda x: x['score'], reverse=False)
    
    json_path = result_dir / "rank.json"
    with open(json_path, "w") as f:
        json.dump(rank_data, f, indent=2)
        
    return rank_data

# --- 5. MAIN ---

def main():
    seed=1; random.seed(seed); torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("input_checkpoint", help="Path to best_model.tar")
    parser.add_argument("output_dir", help="Root output directory")
    parser.add_argument("--backbone", type=str, choices=['b4', 'b5'], default='b5')
    parser.add_argument("--quantize-bits", type=int, choices=[8, 16, 32], default=8)
    parser.add_argument("--prune-amount", type=float, default=0.0)
    parser.add_argument("--dataset", default="CDF")
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--baseline-cache", default="baseline_cache.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--regenerate-heatmaps", action="store_true", help="Force regeneration of heatmaps even if they exist")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Path Resolution
    root_output = Path(args.output_dir).resolve()
    fsbi_root = root_output / "fsbi" / args.backbone
    results_root = fsbi_root / "results"
    analytics_dir = fsbi_root / "analytics"
    
    # Folder Naming
    p_int = int(args.prune_amount * 100)
    if p_int > 0: comp_folder_name = f"q{args.quantize_bits}-p{p_int}"
    else: comp_folder_name = f"q{args.quantize_bits}"
        
    baseline_dir = results_root / "baseline"
    compressed_dir = results_root / comp_folder_name

    _ensure_dir(baseline_dir)
    _ensure_dir(compressed_dir)
    _ensure_dir(analytics_dir)

    print(f"--- Experiment: {comp_folder_name} ---")
    
    # Models
    DetectorClass = get_detector_class(args.backbone)
    baseline_state = load_checkpoint_state(args.input_checkpoint)
    if any(is_quantized_entry(v) for v in baseline_state.values()):
        baseline_state = decompress_state_dict(baseline_state)

    base_model = DetectorClass()
    base_model.load_state_dict(baseline_state, strict=False)
    base_model = base_model.to(device)
    base_size = _state_dict_nbytes(base_model.state_dict()) / (1024**2)

    if not (baseline_dir / "baseline.pth").exists():
        torch.save({"state_dict": base_model.state_dict(), "meta": vars(args)}, baseline_dir / "baseline.pth")

    model_comp = DetectorClass()
    model_comp.load_state_dict(base_model.state_dict(), strict=False)
    save_pth = compressed_dir / f"model.pth"
    comp_size = 0.0
    
    if save_pth.exists():
        print(f"Loading compressed model: {save_pth}")
        loaded = torch.load(save_pth, map_location="cpu")
        comp_state = loaded["state_dict"]
        comp_size = os.path.getsize(save_pth) / (1024**2)
    else:
        print("Compressing...")
        apply_global_pruning(model_comp, args.prune_amount)
        comp_state, _ = quantize_state_dict(model_comp.state_dict(), args.quantize_bits)
        torch.save({"state_dict": comp_state, "meta": vars(args)}, save_pth)
        comp_size = os.path.getsize(save_pth) / (1024**2)

    # Init JSON
    json_path = analytics_dir / "analytics.json"
    analytics_json = {"compression": { "backbone": args.backbone }, "evaluation": {}}
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                ex = json.load(f)
                if "evaluation" in ex: analytics_json["evaluation"] = ex["evaluation"]
        except: pass

    # --- BASELINE RUN (Always check if complete) ---
    base_csv = baseline_dir / "result" / "predictions.csv"
    base_res = load_predictions_from_csv(base_csv)
    
    if base_res is None:
        base_res = evaluate_model(base_model, args.dataset, device, args.n_frames, args.max_samples)
    
    if base_res:
        print("Analytics: Baseline")
        # Run analytics (Images gen will skip if exists unless --regenerate-heatmaps)
        run_analytics_full(base_res["predictions"], base_model, device, baseline_dir, regenerate_heatmaps=args.regenerate_heatmaps)
        
        with open(baseline_dir / "result" / "rank.json", 'r') as f: ranks = json.load(f)
        
        analytics_json["evaluation"]["baseline"] = {
            "model_mb": base_size,
            "auc": base_res["auc"],
            "threshold": 0.5,
            "result": {k: v[:10] for k,v in ranks.items()}
        }
        with open(json_path, "w") as f: json.dump(analytics_json, f, indent=2)

    # --- COMPRESSED RUN ---
    print(f"Analytics: {comp_folder_name}")
    comp_csv = compressed_dir / "result" / "predictions.csv"
    comp_res = load_predictions_from_csv(comp_csv)
    
    model_comp_eval = DetectorClass().to(device)
    model_comp_eval.load_state_dict(decompress_state_dict(comp_state), strict=False)

    if comp_res is None:
        comp_res = evaluate_model(model_comp_eval, args.dataset, device, args.n_frames, args.max_samples)
    
    if comp_res:
        run_analytics_full(comp_res["predictions"], model_comp_eval, device, compressed_dir, regenerate_heatmaps=args.regenerate_heatmaps)
        
        with open(compressed_dir / "result" / "rank.json", 'r') as f: ranks = json.load(f)
        
        analytics_json["evaluation"][comp_folder_name] = {
            "model_mb": comp_size,
            "auc": comp_res["auc"],
            "threshold": 0.5,
            "result": {k: v[:10] for k,v in ranks.items()}
        }

    with open(json_path, "w") as f:
        json.dump(analytics_json, f, indent=2)
    
    print(f"\nSuccess! Analytics: {json_path}")

if __name__ == "__main__":
    main()