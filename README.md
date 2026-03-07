# MC1 Signals Project

This repository contains work for FHNW `gbsv` Mini-Challenge 1 (1D signal processing), using Turkiye, Istanbul-centered audio material as the main context.


## Folder Structure

```text
mc1_signals/
    data/
        raw/
        processed/
        external/
        figures/
    notebooks/
        01_sampling_theorem.ipynb
        02_correlation.ipynb                # work in progress
        03_convolution_deconvolution.ipynb  # work in progress
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
    README.md
```

## Current Status

- `01_sampling_theorem.ipynb`: active and exportable
- `02_correlation.ipynb`: draft/work in progress
- `03_convolution_deconvolution.ipynb`: draft/work in progress

## Setup

From `mc1_signals/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Main source file:

- `data/raw/istanbul_dinliyorum.wav`

Optional context image:

- `data/figures/istanbul.png`


## Outputs

Generated artifacts are written to:

- `outputs/figures/`
- `outputs/audio/`
- `outputs/tables/`
