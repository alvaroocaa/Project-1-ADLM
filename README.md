# Project-1-ADLM
## Object Localization and Classification of Traffic Signals

This repository contains a CNN project for traffic signal/sign classification (and localization, if implemented) using **transfer learning with MobileNetV2**.

## Files
- **`P1.ipynb`** — Executed Jupyter Notebook with the full workflow (training/evaluation/prediction).
- **`traffic_sign_model_final.keras`** — Trained **full model** (architecture + weights), ready for inference.
- **`traffic_sign_weights_final.weights.h5`** — Trained **weights only** (requires the same model architecture before loading).

## Requirements

### Built-in (no installation needed)
- `os`
- `re`
- `time`
- `json`
- `pathlib`
- `pickle`

### External (install with pip)
- `Pillow` *(for `PIL.Image`, `PIL.ImageFont`, `PIL.ImageDraw`)*
- `numpy`
- `tensorflow`
- `matplotlib`
- `tensorflow-datasets` *(imported as `tensorflow_datasets`)*
- `kagglehub`
- `pandas`

### Installation
```bash
pip install pillow numpy tensorflow matplotlib tensorflow-datasets kagglehub pandas
```bash

## Usage

1. Download/clone the repository.
2. Install the required libraries.
3. Open `P1.ipynb` in Jupyter Notebook/JupyterLab.
4. Run the notebook or use the saved model files for inference.

## Notes

- Make sure your Python version is compatible with TensorFlow.
- If using `traffic_sign_weights_final.weights.h5`, define the same model architecture before loading weights.
