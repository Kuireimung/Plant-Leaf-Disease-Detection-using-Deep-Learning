from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from resnet_model import build_resnet18_classifier, build_resnet50_classifier


PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "splits" / "label_map.json"
DEFAULT_MODEL_NAME = "resnet18"
DEFAULT_NUM_CLASSES = 15
DEFAULT_DROPOUT = 0.2
DEFAULT_IMAGE_SIZE = 224
CHECKPOINT_ENV_VAR = "MODEL_CHECKPOINT"

DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Bell Pepper - Bacterial Spot",
    "Pepper__bell___healthy": "Bell Pepper - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato_Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato_Early_blight": "Tomato - Early Blight",
    "Tomato_Late_blight": "Tomato - Late Blight",
    "Tomato_Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato - Spider Mites",
    "Tomato__Target_Spot": "Tomato - Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato - Mosaic Virus",
    "Tomato_healthy": "Tomato - Healthy",
}

CHECKPOINT_HELP = (
    "Copy your trained checkpoint into "
    "models/resnet18_baseline_best.pt "
    "or set the MODEL_CHECKPOINT environment variable to the full file path."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RuntimeModel:
    model: nn.Module
    device: torch.device
    class_names: list[str]
    checkpoint_path: Path


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    top_predictions: Dict[str, float]
    notes: str


def _candidate_checkpoints() -> Iterable[Path]:
    env_path = os.getenv(CHECKPOINT_ENV_VAR)
    if env_path:
        yield Path(env_path).expanduser()

    yield PROJECT_ROOT / "models" / "resnet18_baseline_best.pt"
    yield PROJECT_ROOT / "reports" / "checkpoints" / "resnet18_baseline_best.pt"
    yield PROJECT_ROOT / "resnet18_baseline_best.pt"

    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        for path in sorted(models_dir.glob("*.pt")):
            yield path
        for path in sorted(models_dir.glob("*.pth")):
            yield path

    reports_ckpt_dir = PROJECT_ROOT / "reports" / "checkpoints"
    if reports_ckpt_dir.exists():
        for path in sorted(reports_ckpt_dir.glob("*.pt")):
            yield path
        for path in sorted(reports_ckpt_dir.glob("*.pth")):
            yield path


def _resolve_checkpoint() -> Path:
    for path in _candidate_checkpoints():
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "No trained checkpoint was found. "
        "Expected one of the standard locations inside this project."
    )


def _load_label_map() -> Dict[str, int]:
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _class_names_from_label_map(label_map: Dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def _build_model(name: str, num_classes: int, dropout: float) -> nn.Module:
    normalized_name = name.strip().lower()

    if normalized_name == "resnet18":
        return build_resnet18_classifier(num_classes=num_classes, dropout=dropout)

    if normalized_name == "resnet50":
        return build_resnet50_classifier(num_classes=num_classes, dropout=dropout)

    raise ValueError(f"Unsupported model name: {name}")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_torch_payload(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_model_state(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def _extract_model_settings(payload, label_map: Dict[str, int]) -> tuple[str, int, float]:
    if not isinstance(payload, dict):
        return DEFAULT_MODEL_NAME, len(label_map), DEFAULT_DROPOUT

    config = payload.get("config") or {}
    model_cfg = config.get("model") or {}

    model_name = model_cfg.get("name", DEFAULT_MODEL_NAME)
    num_classes = int(payload.get("num_classes") or model_cfg.get("num_classes") or len(label_map))
    dropout = float(model_cfg.get("dropout", DEFAULT_DROPOUT))
    return model_name, num_classes, dropout


def _resize_shorter_side(image: Image.Image, target_shorter_side: int = 256) -> Image.Image:
    width, height = image.size

    if width <= height:
        new_width = target_shorter_side
        new_height = int(round(height * target_shorter_side / width))
    else:
        new_height = target_shorter_side
        new_width = int(round(width * target_shorter_side / height))

    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return image.resize((new_width, new_height), resample=resample)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))


def _build_eval_tensor(image: Image.Image) -> torch.Tensor:
    image = _resize_shorter_side(image.convert("RGB"), target_shorter_side=256)
    image = _center_crop(image, DEFAULT_IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _display_name(raw_label: str) -> str:
    if raw_label in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw_label]
    return raw_label.replace("___", " - ").replace("__", " ").replace("_", " ").strip()


@lru_cache(maxsize=1)
def load_runtime_model() -> RuntimeModel:
    checkpoint_path = _resolve_checkpoint()
    label_map = _load_label_map()
    class_names = _class_names_from_label_map(label_map)
    device = _get_device()

    payload = _load_torch_payload(checkpoint_path, device)
    model_name, num_classes, dropout = _extract_model_settings(payload, label_map)
    model = _build_model(model_name, num_classes, dropout)
    model.load_state_dict(_extract_model_state(payload), strict=True)
    model.to(device)
    model.eval()

    return RuntimeModel(
        model=model,
        device=device,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
    )


def get_startup_message() -> str:
    return str(get_runtime_status()["message"])


def get_runtime_status() -> Dict[str, object]:
    try:
        checkpoint_path = _resolve_checkpoint()
        return {
            "ready": True,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "message": (
                f"Checkpoint detected: {checkpoint_path.name}. "
                "The model will load lazily on the first prediction."
            ),
        }
    except FileNotFoundError:
        return {
            "ready": False,
            "checkpoint_name": None,
            "checkpoint_path": None,
            "message": (
                "No checkpoint has been detected yet. "
                "The web app is ready, but predictions will start working only after you add the trained .pt file."
            ),
        }


@torch.inference_mode()
def predict_leaf_disease(image: Image.Image) -> PredictionResult:
    runtime = load_runtime_model()
    transformed = _build_eval_tensor(image).unsqueeze(0).to(runtime.device)

    logits = runtime.model(transformed)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    top_values, top_indices = torch.topk(probabilities, k=min(3, len(runtime.class_names)))

    best_index = int(top_indices[0].item())
    raw_label = runtime.class_names[best_index]
    best_label = _display_name(raw_label)
    best_confidence = float(top_values[0].item())

    top_predictions = {
        _display_name(runtime.class_names[int(idx.item())]): float(score.item())
        for score, idx in zip(top_values, top_indices)
    }

    if "healthy" in raw_label.lower():
        notes = (
            f"Model checkpoint: {runtime.checkpoint_path.name}. "
            f"Prediction suggests {best_label}. The leaf appears healthy according to the model."
        )
    else:
        notes = (
            f"Model checkpoint: {runtime.checkpoint_path.name}. "
            f"Prediction suggests {best_label}. Treat this as a model-assisted result and verify with field guidance if needed."
        )

    return PredictionResult(
        label=best_label,
        confidence=best_confidence,
        top_predictions=top_predictions,
        notes=notes,
    )
