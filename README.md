# Crop Disease Detector

This project is a single FastAPI web app for crop disease detection using your trained PlantVillage model.

## Local setup

Create and activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the FastAPI web app:

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Download model from Kaggle

Create a fresh Kaggle API token and place it in:

```text
.kaggle/kaggle.json
```

Then download the notebook outputs into the `models` folder with:

```powershell
$env:KAGGLE_CONFIG_DIR=".kaggle"
.\.venv\Scripts\python.exe -m kaggle kernels output kuireimungkhaleng/crop-disease-detection -p ".\models" -o
```

## Model checkpoint

Copy your trained checkpoint from Kaggle into one of these locations:

- `models/resnet18_baseline_best.pt`
- `reports/checkpoints/resnet18_baseline_best.pt`

You can also point the app to another path with:

```powershell
$env:MODEL_CHECKPOINT="C:\full\path\to\your\checkpoint.pt"
```

The app expects the PlantVillage ResNet-18 training format used by the original notebook and reference repo.

## App files

- FastAPI backend in `app.py`
- HTML template in `templates/index.html`
- Frontend behavior in `static/app.js`
- Styling in `static/styles.css`
