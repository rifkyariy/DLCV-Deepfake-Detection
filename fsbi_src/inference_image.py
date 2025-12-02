"""Image-level inference pipeline with detailed step logging and visualization.

This script detects faces with RetinaFace, crops them to 380x380, feeds them
through the FSBI detector, captures per-layer activations, and produces a
step-by-step Markdown report along with visual artifacts for every stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pywt
import torch
import torch.nn.functional as F
from PIL import Image
from retinaface.pre_trained_models import get_model

from model import Detector
from utils.initialize import init_cdf


TARGET_FACE_SIZE = 380
CLASS_LABELS = ["real", "fake"]
HEATMAP_BLUR_RADIUS = 15
HEATMAP_ALPHA = 0.5


def register_gradcam_hooks(model: Detector):
	captures: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
	handles: List[torch.utils.hooks.RemovableHandle] = []
	layer_order: List[str] = []

	def _record(name: str, module: torch.nn.Module) -> None:
		if module is None:
			return
		layer_order.append(name)
		captures[name] = {"activation": None, "gradient": None}

		def forward_hook(_module, _inp, output):
			out_tensor = output[0] if isinstance(output, tuple) else output
			captures[name]["activation"] = out_tensor.detach().cpu()

		def backward_hook(_module, _grad_in, grad_out):
			grad_tensor = grad_out[0] if isinstance(grad_out, tuple) else grad_out
			if grad_tensor is None:
				return
			captures[name]["gradient"] = grad_tensor.detach().cpu()

		handles.append(module.register_forward_hook(forward_hook))
		handles.append(module.register_full_backward_hook(backward_hook))

	_record("conv_stem", model.net._conv_stem)
	_record("bn0", model.net._bn0)
	for idx, block in enumerate(model.net._blocks):
		_record(f"block_{idx:02d}", block)
	_record("conv_head", model.net._conv_head)
	_record("bn1", model.net._bn1)
	_record("avg_pool", model.net._avg_pooling)
	_record("dropout", model.net._dropout)
	_record("fc", model.net._fc)

	return captures, handles, layer_order


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="FSBI single-image inference")
	parser.add_argument("-i", "--image", help="Path to the input image")
	parser.add_argument(
		"-w",
		"--weights",
		required=True,
		help="Path to the trained checkpoint that contains the 'model' state_dict",
	)
	parser.add_argument(
		"-d",
		"--dataset",
		choices=["cdf"],
		help="Dataset shortcut; when set, run on random samples instead of a single image",
	)
	parser.add_argument(
		"--dataset-phase",
		default="test",
		help="Phase flag passed to dataset initializer when using --dataset",
	)
	parser.add_argument(
		"--sample-count",
		type=int,
		default=5,
		help="Number of random dataset samples to run when --dataset is provided",
	)
	parser.add_argument(
		"--process-name",
		default="fsbi_inference",
		help="Identifier prefix for the output directory",
	)
	parser.add_argument(
		"--output-root",
		default="output",
		help="Root folder where the structured results will be written",
	)
	parser.add_argument(
		"--device",
		default="cuda",
		help="Device to run inference on (cuda, cuda:0, cpu, mps, ...)",
	)
	parser.add_argument(
		"--cam-target",
		choices=["fake", "real", "pred"],
		default="fake",
		help="Which class to backprop for Grad-CAM overlays (default: fake for parity with compression pipeline)",
	)
	args = parser.parse_args()

	if bool(args.image) == bool(args.dataset):
		parser.error("Provide either --image or --dataset (exactly one)")
	if args.dataset and args.sample_count <= 0:
		parser.error("--sample-count must be positive")

	return args

def sample_dataset_images(dataset_name: str, sample_count: int, phase: str) -> List[str]:
	dataset_name = dataset_name.lower()
	if dataset_name == "cdf":
		image_list, _ = init_cdf(phase=phase, level="frame", n_frames=1)
	else:
		raise ValueError(f"Unsupported dataset shortcut: {dataset_name}")

	if not image_list:
		raise RuntimeError(f"Dataset '{dataset_name}' returned no samples. Ensure preprocessing is complete.")

	if len(image_list) <= sample_count:
		return image_list

	return random.sample(image_list, sample_count)


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def setup_device(device_str: str) -> torch.device:
	if device_str.lower().startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available")
	if device_str == "mps" and not torch.backends.mps.is_available():
		raise RuntimeError("MPS requested but not available on this machine")
	return torch.device(device_str)


def load_detector(weights_path: str, device: torch.device) -> Detector:
	model = Detector().to(device)
	checkpoint = torch.load(weights_path, map_location=device)
	state_dict = checkpoint.get("model", checkpoint)
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	return model


def init_face_detector(image_rgb: np.ndarray, device: torch.device):
	max_side = int(max(image_rgb.shape[:2]))
	face_detector = get_model("resnet50_2020-07-20", max_size=max_side, device=device)
	face_detector.eval()
	return face_detector


def prepare_output_dirs(output_root: str, process_name: str, image_path: Path) -> Dict[str, Path]:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	image_id = image_path.stem
	process_dir = Path(output_root) / f"{process_name}_{image_id}_{timestamp}"
	steps_dir = ensure_dir(process_dir / "result" / "steps")

	dirs = {
		"process": process_dir,
		"steps": steps_dir,
		"input": ensure_dir(steps_dir / "a_input"),
		"preprocess": ensure_dir(steps_dir / "b_preprocess"),
		"layers_root": ensure_dir(steps_dir / "c_layers"),
		"classifier": ensure_dir(steps_dir / "d_classifier"),
		"report": process_dir / "result" / "report.md",
		"log": process_dir / "result" / "process.log",
	}
	return dirs


def append_log(log_path: Path, message: str) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	with log_path.open("a", encoding="utf-8") as log_file:
		log_file.write(f"[{timestamp}] {message}\n")


def save_rgb_image(path: Path, rgb_image: np.ndarray) -> None:
	Image.fromarray(rgb_image).save(path)


def detect_primary_face(image_rgb: np.ndarray, face_detector) -> Dict:
	detections = face_detector.predict_jsons(image_rgb)
	if not detections:
		raise RuntimeError("No faces detected in the input image")
	return max(detections, key=lambda det: (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]))


def draw_detection(image_rgb: np.ndarray, detection: Dict) -> np.ndarray:
	vis_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
	x0, y0, x1, y1 = map(int, detection["bbox"])
	cv2.rectangle(vis_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
	for point in detection.get("landmarks", []):
		px, py = map(int, point)
		cv2.circle(vis_bgr, (px, py), 2, (0, 0, 255), -1)
	return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def crop_face(image_rgb: np.ndarray, bbox: List[float], target_size: int = TARGET_FACE_SIZE, margin: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
	h, w = image_rgb.shape[:2]
	x0, y0, x1, y1 = bbox
	box_w = x1 - x0
	box_h = y1 - y0
	side = max(box_w, box_h) * (1 + margin)
	cx = x0 + box_w / 2
	cy = y0 + box_h / 2
	half = side / 2

	x_start = int(max(0, cx - half))
	x_end = int(min(w, cx + half))
	y_start = int(max(0, cy - half))
	y_end = int(min(h, cy + half))

	crop = image_rgb[y_start:y_end, x_start:x_end]
	if crop.size == 0:
		raise RuntimeError("Face crop was empty; check detection coordinates")
	crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
	return crop_resized, (x_start, y_start, x_end, y_end)

def build_face_mask(
	crop_shape: Tuple[int, int, int],
	landmarks: Optional[List[List[float]]],
	crop_window: Tuple[int, int, int, int],
) -> np.ndarray:
	"""Generate a landmark-based mask visualization mirroring SBI mask augmentation."""
	mask = np.zeros((crop_shape[0], crop_shape[1]), dtype=np.uint8)

	if landmarks:
		x_start, y_start, _, _ = crop_window
		points = []
		for point in landmarks:
			x, y = point
			points.append([int(round(x - x_start)), int(round(y - y_start))])
		pts = np.array(points, dtype=np.int32)
		pts[:, 0] = np.clip(pts[:, 0], 0, crop_shape[1] - 1)
		pts[:, 1] = np.clip(pts[:, 1], 0, crop_shape[0] - 1)
		if len(pts) >= 3:
			hull = cv2.convexHull(pts)
			cv2.fillConvexPoly(mask, hull, 255)
		elif len(pts) == 2:
			cv2.line(mask, tuple(pts[0]), tuple(pts[1]), 255, thickness=20)
		elif len(pts) == 1:
			cv2.circle(mask, tuple(pts[0]), 20, 255, -1)
	else:
		h, w = mask.shape
		center = (w // 2, h // 2)
		size = (int(w * 0.35), int(h * 0.5))
		cv2.ellipse(mask, center, size, 0, 0, 360, 255, -1)

	return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

def preprocess_face_tensor(face_rgb: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, Dict, Dict[str, np.ndarray]]:
	"""Apply DWT preprocessing to match training pipeline."""
	# Normalize to 0-1
	face_norm = face_rgb.astype(np.float32) / 255.0
	
	# Split channels
	b, g, r = cv2.split(face_norm)
	
	# Apply DWT to each channel
	cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(r, 'sym2', mode='reflect')
	cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, 'sym2', mode='reflect')
	cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, 'sym2', mode='reflect')
	
	# Resize approximation coefficients to target size
	cA_r = cv2.resize(cA_r, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	cA_g = cv2.resize(cA_g, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	cA_b = cv2.resize(cA_b, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	
	# Resize original channels
	r = cv2.resize(r, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	g = cv2.resize(g, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	b = cv2.resize(b, (TARGET_FACE_SIZE, TARGET_FACE_SIZE), interpolation=cv2.INTER_LINEAR)
	
	# Blend DWT coefficients with original (matching compression pipeline)
	face_dwt = np.array([(cA_r + r) / 2, (cA_g + g) / 2, (cA_b + b) / 2], dtype=np.float32)
	
	# Convert to tensor
	tensor = torch.from_numpy(face_dwt).unsqueeze(0).to(device)
	
	# Create visualization helpers
	dwt_vis = np.transpose(face_dwt, (1, 2, 0))
	dwt_vis_uint8 = np.clip(dwt_vis * 255, 0, 255).astype(np.uint8)

	def _tinted_visual(channel_data: np.ndarray, channel_index: int) -> np.ndarray:
		clamped = np.clip(channel_data, 0.0, 1.0)
		gray = np.uint8(clamped * 255)
		vis = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
		vis[..., channel_index] = gray
		return vis

	dwt_visuals = {
		"combined": dwt_vis_uint8,
		"red": _tinted_visual(face_dwt[0], 0),
		"green": _tinted_visual(face_dwt[1], 1),
		"blue": _tinted_visual(face_dwt[2], 2),
	}
	
	stats = {
		"mean": float(face_dwt.mean()),
		"std": float(face_dwt.std()),
		"min": float(face_dwt.min()),
		"max": float(face_dwt.max()),
		"dwt_coefficients": {
			"cA_r": {"mean": float(cA_r.mean()), "std": float(cA_r.std())},
			"cA_g": {"mean": float(cA_g.mean()), "std": float(cA_g.std())},
			"cA_b": {"mean": float(cA_b.mean()), "std": float(cA_b.std())},
		},
	}
	return tensor, stats, dwt_visuals


def tensor_to_activation_map(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
	data = tensor.float()

	if data.dim() == 4:
		data = data[:1]  # batch first
		data = data.mean(dim=1, keepdim=True)
	elif data.dim() == 3:
		data = data.unsqueeze(0)
		data = data.mean(dim=1, keepdim=True)
	elif data.dim() in (1, 2):
		vec = data.view(-1)
		side = math.ceil(math.sqrt(vec.numel()))
		padded = torch.zeros(side * side, dtype=vec.dtype)
		padded[: vec.numel()] = vec
		data = padded.view(1, 1, side, side)
	else:
		raise ValueError(f"Unsupported tensor shape for visualization: {tuple(data.shape)}")

	data = data - data.min()
	data = data / (data.max() + 1e-8)
	data = F.interpolate(data, size=target_size, mode="bilinear", align_corners=False)
	return data.squeeze(0).squeeze(0)


def compute_layer_cam(
	activation: torch.Tensor,
	gradient: torch.Tensor,
	target_size: Tuple[int, int],
) -> np.ndarray:
	act = activation.clone().float()
	grad = gradient.clone().float()

	if act.dim() == 4:
		act = act[:1]
		grad = grad[:1]
		weights = grad.mean(dim=(2, 3), keepdim=True)
		cam = torch.sum(weights * act, dim=1, keepdim=True)
		cam = F.relu(cam)
		cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
		cam_map = cam.squeeze(0).squeeze(0).cpu().numpy()
	else:
		if grad.shape == act.shape:
			combined = act * grad
		else:
			combined = act
		cam_tensor = tensor_to_activation_map(combined, target_size)
		cam_map = cam_tensor.cpu().numpy()

	cam_min, cam_max = cam_map.min(), cam_map.max()
	if cam_max - cam_min > 1e-8:
		cam_map = (cam_map - cam_min) / (cam_max - cam_min)
	else:
		cam_map = np.zeros_like(cam_map)

	return cam_map


def build_layer_visuals(
	captures: Dict[str, Dict[str, Optional[torch.Tensor]]],
	layer_order: List[str],
	base_image_rgb: np.ndarray,
	layers_root: Path,
) -> List[Dict]:
	summaries: List[Dict] = []
	target_hw = (base_image_rgb.shape[0], base_image_rgb.shape[1])
	base_bgr = cv2.cvtColor(base_image_rgb, cv2.COLOR_RGB2BGR)

	for name in layer_order:
		record = captures.get(name)
		if not record:
			continue
		activation = record.get("activation")
		gradient = record.get("gradient")
		if activation is None or gradient is None:
			continue

		layer_dir = ensure_dir(layers_root / name)
		image_dir = ensure_dir(layer_dir / "image")
		heatmap_dir = ensure_dir(layer_dir / "heatmap")

		cam_map = compute_layer_cam(activation, gradient, target_hw).astype(np.float32)
		if HEATMAP_BLUR_RADIUS > 0:
			radius = HEATMAP_BLUR_RADIUS if HEATMAP_BLUR_RADIUS % 2 == 1 else HEATMAP_BLUR_RADIUS + 1
			cam_map = cv2.GaussianBlur(cam_map, (radius, radius), 0)
			cam_min, cam_max = cam_map.min(), cam_map.max()
			if cam_max - cam_min > 1e-8:
				cam_map = (cam_map - cam_min) / (cam_max - cam_min)

		activation_gray = np.uint8(cam_map * 255)

		activation_color = cv2.applyColorMap(activation_gray, cv2.COLORMAP_VIRIDIS)
		activation_rgb = cv2.cvtColor(activation_color, cv2.COLOR_BGR2RGB)

		heatmap_color = cv2.applyColorMap(activation_gray, cv2.COLORMAP_JET)
		overlay_bgr = cv2.addWeighted(base_bgr, 1.0 - HEATMAP_ALPHA, heatmap_color, HEATMAP_ALPHA, 0)
		overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

		activation_path = image_dir / "activation.png"
		heatmap_path = heatmap_dir / "overlay.png"
		save_rgb_image(activation_path, activation_rgb)
		save_rgb_image(heatmap_path, overlay_rgb)

		stats = {
			"activation_shape": tuple(int(dim) for dim in activation.shape),
			"gradient_shape": tuple(int(dim) for dim in gradient.shape),
			"cam_min": float(cam_map.min()),
			"cam_max": float(cam_map.max()),
			"cam_mean": float(cam_map.mean()),
			"cam_std": float(cam_map.std()),
		}

		with (layer_dir / "stats.json").open("w", encoding="utf-8") as stats_file:
			json.dump(stats, stats_file, indent=2)

		summaries.append(
			{
				"name": name,
				"shape": stats["activation_shape"],
				"activation_image": activation_path,
				"heatmap_image": heatmap_path,
				"stats": stats,
			}
		)

	return summaries



def format_rel_path(path: Path | None, base_dir: Path) -> str:
	return os.path.relpath(path, base_dir) if path else "-"


def write_markdown(report_path: Path, context: Dict) -> None:
	report_dir = context.get("report_dir", report_path.parent)
	rel = lambda path: format_rel_path(path, report_dir)
	md_lines = [
		f"# Inference Report — {context['image_id']}",
		"",
		"## Overview",
		f"- **Process ID:** {context['process_id']}",
		f"- **Timestamp:** {context['timestamp']}",
		f"- **Device:** {context['device']}",
		f"- **Weights:** {context['weights']}",
		f"- **Input Image:** {context['input_path']}",
		f"- **Grad-CAM target:** {context.get('cam_target')} (index {context.get('cam_target_index')})",
		"",
		"## Step A — Input Image",
		f"- Dimensions: {context['input_shape']} (HxW)",
		"",
		f"![Original input]({rel(context['input_image_saved'])})",
		"",
		f"![Detection overlay]({rel(context['detection_overlay'])})",
		"",
		"## Step B — Face Detection",
		"",
		"1. **Source/Target Generator (SBI):** During training, the base frame is duplicated into target and source views. Augmentations applied to either branch mimic compression, color, and motion drift so the model sees realistic perturbations.",
		"2. **Mask Generator:** Landmarks and RetinaFace detections create a convex-hull face mask (or `random_get_hull` when available). Additional affine + elastic warps make every mask unique.",
		"3. **Blending (Self-Blended Image):** The source face is composited onto the target face through the generated mask using `dynamic_blend`, yielding an SBI pair (fake vs real) that the detector learns from.",
		"",
		"### Runtime Face Detection Snapshot",
		f"- Bounding box (x0, y0, x1, y1): {context['bbox']}",
		f"- Landmarks (x, y): {context['landmarks']}",
		"",
		f"![Cropped face]({rel(context['crop_image'])})",
		"",
		"### Landmark Mask Approximation",
		f"![Mask only]({rel(context['mask_image'])})",
		"",
		"## Step C — Frequency Features (DWT/FFG)",
		"",
		"### Channel-Wise Wavelet Pass (sym2, reflect)",
		"a. **DWT from R:** Extract approximation map `cA_r`, resize, blend with original red channel `(cA_r + R) / 2`.",
		"b. **DWT from G:** Same pipeline for the green channel producing `cA_g`.",
		"c. **DWT from B:** Blue channel variant yielding `cA_b`.",
		"d. **Combine as Channels:** Stack `[cA_r, cA_g, cA_b]` to form the FSBI tensor fed to EfficientNet-B5.",
		"",
		"### Visual Comparison",
		f"![Original face]({rel(context['crop_image'])})",
		"",
		f"![Combined channels]({rel(context['dwt_image'])})",
		"",
		"### Channel Breakdown",
		"Channel | Visualization",
		"---|---",
		f"Red (cA_r) | ![]({rel(context['dwt_channels']['red'])})",
		f"Green (cA_g) | ![]({rel(context['dwt_channels']['green'])})",
		f"Blue (cA_b) | ![]({rel(context['dwt_channels']['blue'])})",
		"",
		"### Statistics",
		f"- **Blended tensor:** mean {context['tensor_stats']['mean']:.4f}, std {context['tensor_stats']['std']:.4f},"
		f" min {context['tensor_stats']['min']:.4f}, max {context['tensor_stats']['max']:.4f}",
		f"- **cA_r:** mean {context['tensor_stats']['dwt_coefficients']['cA_r']['mean']:.4f}, std {context['tensor_stats']['dwt_coefficients']['cA_r']['std']:.4f}",
		f"- **cA_g:** mean {context['tensor_stats']['dwt_coefficients']['cA_g']['mean']:.4f}, std {context['tensor_stats']['dwt_coefficients']['cA_g']['std']:.4f}",
		f"- **cA_b:** mean {context['tensor_stats']['dwt_coefficients']['cA_b']['mean']:.4f}, std {context['tensor_stats']['dwt_coefficients']['cA_b']['std']:.4f}",
		"",
		"## Step D — Feature Journey",
		"Layer | Shape | Activation | Heatmap",
		"---|---|---|---",
	]

	for layer in context["layers"]:
		activation_rel = rel(layer["activation_image"])
		heatmap_rel = rel(layer["heatmap_image"])
		md_lines.append(
			f"{layer['name']} | {layer['shape']} | ![]({activation_rel}) | ![]({heatmap_rel})"
		)

	md_lines.extend(
		[
			"",
			"## Step E — Classifier",
			f"- Logits: {context['logits']}",
			f"- Probabilities: {context['probabilities']}",
			f"- Predicted label: **{context['prediction']}**",
			f"- Class confidence: {context['confidence']:.4f}",
		]
	)

	ensure_dir(report_path.parent)
	with report_path.open("w", encoding="utf-8") as report_file:
		report_file.write("\n".join(md_lines))


def run_pipeline(args: argparse.Namespace) -> Path:
	image_path = Path(args.image)
	if not image_path.exists():
		raise FileNotFoundError(f"Input image not found: {image_path}")

	device = setup_device(args.device)
	append_log(Path(args.output_root) / "inference.log", f"Launching run for {image_path}")

	image_bgr = cv2.imread(str(image_path))
	if image_bgr is None:
		raise RuntimeError(f"Failed to read image: {image_path}")
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

	paths = prepare_output_dirs(args.output_root, args.process_name, image_path)
	append_log(paths["log"], "Initialized output directories")

	save_rgb_image(paths["input"] / "original.png", image_rgb)

	detector_model = load_detector(args.weights, device)
	face_detector = init_face_detector(image_rgb, device)
	append_log(paths["log"], "Models loaded")

	detection = detect_primary_face(image_rgb, face_detector)
	detection_overlay = draw_detection(image_rgb, detection)
	save_rgb_image(paths["input"] / "retinaface_overlay.png", detection_overlay)
	append_log(paths["log"], "Face detected and logged")

	cropped_rgb, crop_window = crop_face(image_rgb, detection["bbox"], TARGET_FACE_SIZE)
	save_rgb_image(paths["preprocess"] / "cropped_face.png", cropped_rgb)

	mask_rgb = build_face_mask(cropped_rgb.shape, detection.get("landmarks"), crop_window)
	save_rgb_image(paths["preprocess"] / "mask.png", mask_rgb)

	face_tensor, tensor_stats, dwt_visuals = preprocess_face_tensor(cropped_rgb, device)
	save_rgb_image(paths["preprocess"] / "dwt_processed.png", dwt_visuals["combined"])
	dwt_channel_paths = {
		"red": paths["preprocess"] / "dwt_red.png",
		"green": paths["preprocess"] / "dwt_green.png",
		"blue": paths["preprocess"] / "dwt_blue.png",
		"combined": paths["preprocess"] / "dwt_processed.png",
	}
	save_rgb_image(dwt_channel_paths["red"], dwt_visuals["red"])
	save_rgb_image(dwt_channel_paths["green"], dwt_visuals["green"])
	save_rgb_image(dwt_channel_paths["blue"], dwt_visuals["blue"])

	captures, handles, layer_order = register_gradcam_hooks(detector_model)
	logits = detector_model(face_tensor)
	probs = torch.softmax(logits.detach(), dim=1)

	logits_list = logits.detach().squeeze(0).tolist()
	probs_list = probs.squeeze(0).tolist()
	prediction_idx = int(np.argmax(probs_list))
	prediction_label = CLASS_LABELS[prediction_idx] if prediction_idx < len(CLASS_LABELS) else str(prediction_idx)

	cam_target_idx_map = {label: idx for idx, label in enumerate(CLASS_LABELS)}
	if args.cam_target == "pred":
		cam_target_idx = prediction_idx
		cam_target_label = prediction_label
	else:
		cam_target_idx = cam_target_idx_map.get(args.cam_target, prediction_idx)
		cam_target_label = args.cam_target

	detector_model.zero_grad(set_to_none=True)
	target_logit = logits[:, cam_target_idx].sum()
	target_logit.backward()

	layer_summaries = build_layer_visuals(captures, layer_order, cropped_rgb, paths["layers_root"])
	for handle in handles:
		handle.remove()

	append_log(paths["log"], f"Captured {len(layer_summaries)} layer visualizations")

	classifier_record = {
		"logits": logits_list,
		"probabilities": probs_list,
		"prediction": prediction_label,
		"confidence": probs_list[prediction_idx],
	}
	with (paths["classifier"] / "classifier.json").open("w", encoding="utf-8") as cls_file:
		json.dump(classifier_record, cls_file, indent=2)

	metadata = {
		"image_id": image_path.stem,
		"process_id": paths["process"].name,
		"timestamp": datetime.now().isoformat(),
		"device": str(device),
		"weights": os.path.abspath(args.weights),
		"input_path": os.path.abspath(image_path),
		"input_shape": f"{image_rgb.shape[0]}x{image_rgb.shape[1]}",
		"input_image_saved": paths["input"] / "original.png",
		"detection_overlay": paths["input"] / "retinaface_overlay.png",
		"bbox": [round(float(coord), 2) for coord in detection["bbox"]],
		"landmarks": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in detection.get("landmarks", [])],
		"crop_window": list(map(int, crop_window)),
		"crop_image": paths["preprocess"] / "cropped_face.png",
		"mask_image": paths["preprocess"] / "mask.png",
		"dwt_image": paths["preprocess"] / "dwt_processed.png",
		"preprocess_image": paths["preprocess"] / "cropped_face.png",
		"tensor_stats": tensor_stats,
		"dwt_channels": dwt_channel_paths,
		"layers": layer_summaries,
		"logits": logits_list,
		"probabilities": probs_list,
		"prediction": prediction_label,
		"confidence": classifier_record["confidence"],
		"process_dir": paths["process"],
		"report_dir": paths["report"].parent,
		"cam_target": cam_target_label,
		"cam_target_index": cam_target_idx,
	}

	write_markdown(paths["report"], metadata)
	append_log(paths["log"], "Markdown report generated")

	return paths["report"]


def main() -> None:
	args = parse_args()

	if args.dataset:
		sample_images = sample_dataset_images(args.dataset, args.sample_count, args.dataset_phase)
		print(f"Running {len(sample_images)} samples from dataset '{args.dataset.upper()}'")
		reports: List[Path] = []
		for idx, sample_path in enumerate(sample_images, start=1):
			print(f"[{idx}/{len(sample_images)}] Processing {sample_path}")
			run_args = argparse.Namespace(**vars(args))
			run_args.image = sample_path
			run_args.dataset = None
			reports.append(run_pipeline(run_args))
		print("\nCompleted dataset run. Reports:")
		for rpt in reports:
			print(f"- {rpt}")
	else:
		report_path = run_pipeline(args)
		print(f"Completed inference. Report saved to: {report_path}")


if __name__ == "__main__":
	main()
