from __future__ import annotations

import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from inference import CHECKPOINT_HELP, get_runtime_status, predict_leaf_disease


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Crop Disease Detector",
    description="Upload a leaf image and predict crop disease classes with trained ResNet or EfficientNet models.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    status = get_runtime_status()
    supported_models = status.get("models", [])
    ready_model_count = sum(1 for model in supported_models if model.get("ready"))
    model_lookup = {
        str(model.get("id")): model
        for model in supported_models
    }
    resnet_status = model_lookup.get("resnet18", {})
    efficientnet_status = model_lookup.get("efficientnet_b0", {})
    context = {
        "request": request,
        "status": status,
        "checkpoint_help": CHECKPOINT_HELP,
        "supported_crops": [
            "Apple",
            "Blueberry",
            "Cherry",
            "Corn",
            "Grape",
            "Orange",
            "Peach",
            "Pepper bell",
            "Potato",
            "Raspberry",
            "Soybean",
            "Squash",
            "Strawberry",
            "Tomato",
        ],
        "models_ready_label": f"{ready_model_count} / {len(supported_models)}",
        "resnet_ready": bool(resnet_status.get("ready")),
        "resnet_checkpoint_name": resnet_status.get("checkpoint_name") or "Checkpoint missing",
        "efficientnet_ready": bool(efficientnet_status.get("ready")),
        "efficientnet_checkpoint_name": efficientnet_status.get("checkpoint_name") or "Checkpoint missing",
    }
    page = templates.get_template("index.html")
    return HTMLResponse(page.render(context))


@app.get("/api/status")
async def api_status() -> dict[str, object]:
    return get_runtime_status()


@app.get("/health")
async def health() -> dict[str, object]:
    status = get_runtime_status()
    return {"status": "ok", "model_ready": status["ready"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("resnet18")) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    try:
        result = predict_leaf_disease(image, model_id=model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"{exc} {CHECKPOINT_HELP}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    top_predictions = [
        {"label": label, "confidence": score}
        for label, score in result.top_predictions.items()
    ]

    return {
        "label": result.label,
        "confidence": result.confidence,
        "model_name": result.model_name,
        "is_supported_image": result.is_supported_image,
        "top_predictions": top_predictions,
        "notes": result.notes,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
