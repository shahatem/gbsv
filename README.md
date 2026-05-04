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

2D image processing with Osman Hamdi Bey's *The Tortoise Trainer* as the primary source image.

### Folder Structure

```text
mc2_images/
    data/
        raw/
            Osman_Hamdi_Bey - The_Tortoise_Trainer.jpg
            the_tortoise_trainer_header.png
        processed/
    notebooks/
        01_augmentation.ipynb
        02_pattern_detection.ipynb
    outputs/
        figures/
        tables/
    src/
        image_utils.py
        plotting_utils.py
        metrics_utils.py
    requirements.txt
```

### Status

| Notebook | Status |
|---|---|
| `01_augmentation.ipynb` | Finished — Pending Feedback Implementation |
| `02_pattern_detection.ipynb` | Finished — Pending Feedback Implementation |

### Setup

From `mc2_images/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
