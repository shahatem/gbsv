# gbsv Türkiye

This repository contains work for FHNW `gbsv` mini-challenges, using Türkiye / Istanbul-centered audio and visual material as the main context.

---

## Mini-Challenge 1 — Signals (`mc1_signals/`)

1D signal processing with Istanbul audio material.

### Folder Structure

```text
mc1_signals/
    data/
        raw/
            istanbul_dinliyorum.wav
        figures/
            istanbul.png
            istanbul_bazar.jpg
            istanbul_hagia_sophia.jpg
    notebooks/
        01_sampling_theorem.ipynb
        02_correlation.ipynb
        03_convolution_deconvolution.ipynb
    outputs/
        figures/
        audio/
        tables/
    src/
        audio_utils.py
        signal_utils.py
        plotting_utils.py
        metrics_utils.py
    requirements.txt
```

### Status

| Notebook | Status |
|---|---|
| `01_sampling_theorem.ipynb` | Finished — Pending Feedback Implementation |
| `02_correlation.ipynb` | Finished — Pending Feedback Implementation |
| `03_convolution_deconvolution.ipynb` | Finished — Pending Feedback Implementation |

### Setup

From `mc1_signals/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Mini-Challenge 2 — Images (`mc2_images/`)

2D image processing with Osman Hamdi Bey's *The Tortoise Trainer* (1906, Pera Museum Istanbul) as the primary source image. The three notebooks form a progressive pipeline: photometric augmentation → structural edge analysis → object-level segmentation.

### Folder Structure

```text
mc2_images/
    data/
        raw/
            Osman_Hamdi_Bey - The_Tortoise_Trainer.jpg   # 960 × 1796 px source
            the_tortoise_trainer_header.png               # cropped header for display
        processed/
    notebooks/
        01_augmentation.ipynb           # contrast / brightness / noise augmentation
        02_pattern_detection.ipynb      # Canny edge detection + parameter sweep
        03_segmentation_object_analysis.ipynb  # Otsu thresholding + connected-component sweep
        01_augmentation.html            # exported HTML
        02_pattern_detection.html
        03_segmentation_object_analysis.html
    outputs/
        figures/                        # all saved plots (PNG)
        tables/
            01_augmentation_sweep.csv
            02_pattern_sweep.csv
            03_segmentation_sweep.csv
    src/
        image_utils.py
        metrics_utils.py
        plotting_utils.py
    requirements.txt
```

### Status

| Notebook | Status |
|---|---|
| `01_augmentation.ipynb` | Finished |
| `02_pattern_detection.ipynb` | Finished |
| `03_segmentation_object_analysis.ipynb` | Finished |

### Setup

From `mc2_images/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
