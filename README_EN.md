# SAM 2 Web UI

An interactive web application for image segmentation using SAM 2 (Segment Anything Model 2).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Interactive Segmentation**: Click on images to instantly run segmentation
- **Multiple Results Display**: Shows the best result and other candidates simultaneously
- **Cutout Image Download**: Download segmentation results as transparent PNG files
- **Boundary Detection Modes**: 3-level (Narrow/Standard/Wide) adjustment for boundary precision
- **Boundary Smoothing**: Gaussian blur and morphological processing for smoother boundaries
- **Custom Threshold**: Detailed threshold adjustment available

## Screenshots

![Screenshot](img/screenshot.png)

The application features a 2-column layout:
- **Left**: Click on images to specify coordinates
- **Right**: Display segmentation results and cutout images

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tomorrow56/sam2-webui.git
cd sam2-webui
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run sam2_webui.py --server.headless true
```

> **Note**: You may be prompted for an email address on first launch. Press Enter to skip. Use the `--server.headless true` option to avoid this prompt.

## Usage

1. Access `http://localhost:8501` in your browser
2. Upload an image from the sidebar (JPG, PNG, BMP supported)
3. Click on the image in the left panel to run segmentation
4. Results will be displayed on the right
5. Use the "Download Cutout Image" button to save results

### Sidebar Settings

#### 🎚️ Segmentation Adjustment
- **Narrow (Precise)**: Detect object boundaries precisely
- **Standard**: Standard boundary detection
- **Wide (Rough)**: Detect objects more broadly

#### 🔧 Detailed Settings
- **Custom Threshold**: Fine-tune from -2.0 to 2.0

#### ✨ Boundary Smoothing
- **Gaussian Blur**: Blur mask boundaries for smoothness
- **Morphology (Open/Close)**: Remove small noise and fill holes
- **Both**: Apply both processes (smoothest)

## File Structure

```
sam2-webui/
├── sam2_webui.py         # Main application
├── requirements.txt      # Dependencies
├── translations.py       # Multilingual support (Japanese/English)
├── README.md            # Japanese documentation
├── README_EN.md         # English documentation
├── LICENSE              # MIT License
├── checkpoints/         # Model files (auto-download)
└── img/                 # Image files (screenshots, etc.)
```

## System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU (recommended) or CPU
- **RAM**: 8GB or more (16GB+ recommended for Large model)

## Models

The following models will be automatically downloaded on first launch:

| Model | Size | Features |
|--------|--------|------|
| sam2.1_hiera_small | ~150MB | Fast, lightweight |
| sam2.1_hiera_base_plus | ~300MB | Balanced |
| sam2.1_hiera_large | ~800MB | High precision |

### How to Change Models

To change the model being used, modify the following parts in `sam2_webui.py`:

```python
# Around line 126 in sam2_webui.py
def download_model():
    model_name = "sam2.1_hiera_small.pt"  # ← Change this
    model_path = f"checkpoints/{model_name}"
    
    # ... (omitted) ...
    
    # Around line 148 in sam2_webui.py
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", model_path, device=device)  # ← Also change this
```

#### Configuration Examples:

**Small model (default)**
```python
model_name = "sam2.1_hiera_small.pt"
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", model_path, device=device)
```

**Base_plus model**
```python
model_name = "sam2.1_hiera_base_plus.pt"
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", model_path, device=device)
```

**Large model**
```python
model_name = "sam2.1_hiera_large.pt"
sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", model_path, device=device)
```

After making changes, restart the application and the new model will be downloaded.

## Dependencies

- streamlit >= 1.28.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- plotly >= 5.0.0
- streamlit-plotly-events >= 0.0.6
- [SAM 2](https://github.com/facebookresearch/sam2)

## Notes

- Models will be downloaded on first launch (may take several minutes)
- GPU environment provides faster performance
- Large images may take longer to process

## License

MIT License

## Acknowledgments

- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) - Meta AI Research
- [Streamlit](https://streamlit.io/) - Web UI framework
