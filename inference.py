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
from torchvision import models
from PIL import Image

from resnet_model import build_resnet18_classifier, build_resnet50_classifier


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_PROJECT_ROOT = PROJECT_ROOT.parent / "plantvillage-disease-classifier"
LABEL_MAP_PATH = PROJECT_ROOT / "data" / "splits" / "label_map.json"
TRAINING_LABEL_MAP_PATH = TRAINING_PROJECT_ROOT / "data" / "splits" / "label_map.json"
DEFAULT_MODEL_NAME = "resnet18"
DEFAULT_NUM_CLASSES = 15
DEFAULT_DROPOUT = 0.2
DEFAULT_IMAGE_SIZE = 224
MIN_SUPPORTED_CONFIDENCE = 0.60
CHECKPOINT_ENV_VAR = "MODEL_CHECKPOINT"
MODEL_ENV_VARS = {
    "resnet18": "RESNET_CHECKPOINT",
    "efficientnet_b0": "EFFICIENTNET_CHECKPOINT",
}
MODEL_REGISTRY = {
    "resnet18": {
        "label": "ResNet-18",
        "description": "Strong baseline CNN with residual blocks.",
        "candidates": [
            TRAINING_PROJECT_ROOT / "reports" / "checkpoints" / "resnet18_baseline_best.pt",
            PROJECT_ROOT / "models" / "resnet18_baseline_best.pt",
            PROJECT_ROOT / "reports" / "checkpoints" / "resnet18_baseline_best.pt",
            PROJECT_ROOT / "resnet18_baseline_best.pt",
        ],
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "description": "Compact transfer model optimized for efficient feature extraction.",
        "candidates": [
            TRAINING_PROJECT_ROOT / "reports" / "checkpoints" / "efficientnet_b0_baseline_best.pt",
            PROJECT_ROOT / "models" / "efficientnet_b0_baseline_best.pt",
            PROJECT_ROOT / "reports" / "checkpoints" / "efficientnet_b0_baseline_best.pt",
            PROJECT_ROOT / "efficientnet_b0_baseline_best.pt",
        ],
    },
}

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
    "Use the trained checkpoints from plantvillage-disease-classifier/reports/checkpoints, "
    "copy checkpoints into this app's models folder, or set RESNET_CHECKPOINT / "
    "EFFICIENTNET_CHECKPOINT to full file paths."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RuntimeModel:
    model: nn.Module
    device: torch.device
    class_names: list[str]
    checkpoint_path: Path
    model_id: str
    display_name: str


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    top_predictions: Dict[str, float]
    notes: str
    model_name: str
    is_supported_image: bool


def _candidate_checkpoints(model_id: str) -> Iterable[Path]:
    env_path = os.getenv(MODEL_ENV_VARS.get(model_id, ""))
    if env_path:
        yield Path(env_path).expanduser()

    env_path = os.getenv(CHECKPOINT_ENV_VAR)
    if env_path:
        yield Path(env_path).expanduser()

    registry_entry = MODEL_REGISTRY.get(model_id)
    if registry_entry:
        yield from registry_entry["candidates"]

    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        for path in sorted(models_dir.glob(f"*{model_id}*.pt")):
            yield path
        for path in sorted(models_dir.glob(f"*{model_id}*.pth")):
            yield path

    reports_ckpt_dir = PROJECT_ROOT / "reports" / "checkpoints"
    if reports_ckpt_dir.exists():
        for path in sorted(reports_ckpt_dir.glob(f"*{model_id}*.pt")):
            yield path
        for path in sorted(reports_ckpt_dir.glob(f"*{model_id}*.pth")):
            yield path


def _resolve_checkpoint(model_id: str = DEFAULT_MODEL_NAME) -> Path:
    normalized_id = _normalize_model_id(model_id)
    for path in _candidate_checkpoints(normalized_id):
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        f"No trained {MODEL_REGISTRY[normalized_id]['label']} checkpoint was found. "
        "Expected one of the standard locations inside this project."
    )


def _load_json_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_label_map_for_num_classes(num_classes: int) -> Dict[str, int]:
    for path in (TRAINING_LABEL_MAP_PATH, LABEL_MAP_PATH):
        if not path.exists():
            continue
        label_map = _load_json_map(path)
        if len(label_map) == num_classes:
            return label_map

    if LABEL_MAP_PATH.exists():
        label_map = _load_json_map(LABEL_MAP_PATH)
        if len(label_map) == num_classes:
            return label_map

    raise ValueError(
        f"No label_map.json matches the checkpoint's {num_classes} output classes."
    )


def _class_names_from_label_map(label_map: Dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def _normalize_model_id(name: str) -> str:
    normalized_name = name.strip().lower().replace("-", "_")
    if normalized_name in {"resnet", "resnet18"}:
        return "resnet18"
    if normalized_name in {"efficientnet", "efficientnet_b0", "effnet", "effnet_b0"}:
        return "efficientnet_b0"
    raise ValueError(f"Unsupported model name: {name}")


def _build_model(name: str, num_classes: int, dropout: float) -> nn.Module:
    normalized_name = _normalize_model_id(name)

    if normalized_name == "resnet18":
        return build_resnet18_classifier(num_classes=num_classes, dropout=dropout)

    if normalized_name == "resnet50":
        return build_resnet50_classifier(num_classes=num_classes, dropout=dropout)

    if normalized_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

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


def _extract_num_classes(payload) -> int:
    if not isinstance(payload, dict):
        return DEFAULT_NUM_CLASSES

    config = payload.get("config") or {}
    model_cfg = config.get("model") or {}
    return int(payload.get("num_classes") or model_cfg.get("num_classes") or DEFAULT_NUM_CLASSES)


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
    readable = raw_label.replace("___", " - ").replace("__", " ")
    readable = readable.replace("(", "").replace(")", "")
    readable = readable.replace(",", "").replace("_", " ")
    return " ".join(readable.split()).strip()


@lru_cache(maxsize=len(MODEL_REGISTRY))
def load_runtime_model(model_id: str = DEFAULT_MODEL_NAME) -> RuntimeModel:
    normalized_id = _normalize_model_id(model_id)
    checkpoint_path = _resolve_checkpoint(normalized_id)
    device = _get_device()
    payload = _load_torch_payload(checkpoint_path, device)

    num_classes = _extract_num_classes(payload)
    label_map = _load_label_map_for_num_classes(num_classes)
    class_names = _class_names_from_label_map(label_map)

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
        model_id=normalized_id,
        display_name=MODEL_REGISTRY[normalized_id]["label"],
    )


def get_startup_message() -> str:
    return str(get_runtime_status()["message"])


def get_runtime_status() -> Dict[str, object]:
    models_status = []
    for model_id, metadata in MODEL_REGISTRY.items():
        try:
            checkpoint_path = _resolve_checkpoint(model_id)
            models_status.append(
                {
                    "id": model_id,
                    "label": metadata["label"],
                    "description": metadata["description"],
                    "ready": True,
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
        except FileNotFoundError:
            models_status.append(
                {
                    "id": model_id,
                    "label": metadata["label"],
                    "description": metadata["description"],
                    "ready": False,
                    "checkpoint_name": None,
                    "checkpoint_path": None,
                }
            )

    ready_models = [item for item in models_status if item["ready"]]
    default_model = next((item for item in models_status if item["id"] == DEFAULT_MODEL_NAME), models_status[0])

    return {
        "ready": bool(ready_models),
        "checkpoint_name": default_model["checkpoint_name"],
        "checkpoint_path": default_model["checkpoint_path"],
        "models": models_status,
        "message": (
            f"{len(ready_models)} of {len(models_status)} model checkpoints detected. "
            "Choose ResNet-18 or EfficientNet-B0 before running a prediction."
            if ready_models
            else (
                "No checkpoint has been detected yet. "
                "The web app is ready, but predictions will start working only after you add trained .pt files."
            )
        ),
    }


def get_supported_models() -> list[Dict[str, object]]:
    try:
        return list(get_runtime_status()["models"])
    except Exception:
        return []


@torch.inference_mode()
def predict_leaf_disease(image: Image.Image, model_id: str = DEFAULT_MODEL_NAME) -> PredictionResult:
    runtime = load_runtime_model(model_id)
    transformed = _build_eval_tensor(image).unsqueeze(0).to(runtime.device)

    logits = runtime.model(transformed)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
    top_values, top_indices = torch.topk(probabilities, k=min(3, len(runtime.class_names)))

    best_index = int(top_indices[0].item())
    raw_label = runtime.class_names[best_index]
    best_label = _display_name(raw_label)
    best_confidence = float(top_values[0].item())
    is_supported_image = best_confidence >= MIN_SUPPORTED_CONFIDENCE

    top_predictions = {
        _display_name(runtime.class_names[int(idx.item())]): float(score.item())
        for score, idx in zip(top_values, top_indices)
    }

    if not is_supported_image:
        best_label = "Not a supported crop leaf image"
        notes = (
            f"Model: {runtime.display_name}. Checkpoint: {runtime.checkpoint_path.name}. "
            "This image is outside the PlantVillage leaf detector's expected input. "
            "Upload a clear photo of a crop leaf; the classes below are only the closest forced matches, not a diagnosis."
        )
    elif "healthy" in raw_label.lower():
        notes = (
            f"Model: {runtime.display_name}. Checkpoint: {runtime.checkpoint_path.name}. "
            f"Prediction suggests {best_label}. The leaf appears healthy according to the model."
        )
    else:
        notes = (
            f"Model: {runtime.display_name}. Checkpoint: {runtime.checkpoint_path.name}. "
            f"Prediction suggests {best_label}. Treat this as a model-assisted result and verify with field guidance if needed."
        )

    return PredictionResult(
        label=best_label,
        confidence=best_confidence,
        top_predictions=top_predictions,
        notes=notes,
        model_name=runtime.display_name,
        is_supported_image=is_supported_image,
    )
