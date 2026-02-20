# mmJEPA-ECG

## 1. Introduction
🚨 **Note: This repository currently provides preliminary code.**

This repository contains the preliminary PyTorch implementation for the paper *"mmJEPA-ECG: Cross-Posture Robust Contactless Electrocardiogram Monitoring via Millimeter Wave Radar Sensing"*. 
## 2. Code Structure
The core code files and directory structure of this repository are as follows:

* **`mmJEPA.py`**: Defines the JEPA encoder model for self-supervised pretraining on radar signals.
* **`custom_model.py`**: Defines the Diffusion Transformer (DiT) model integrated with Hierarchical Radar Conditioning (HRC).
* **`inference_from_train.py`**: Script for model inference, generation sampling, and evaluation.
* **`block/` directory**: Contains foundational network components, such as attention mechanisms (`attention.py`), positional embeddings (`Positionemb.py`), data loading (`data_load.py`), and backbone construction utilities.
* **`diffusion/` directory**: Contains the core algorithm and sampling logic for the diffusion model (e.g., `gaussian_diffusion.py`, `respace.py`, etc.).

## 3. Dataset Statement
To protect subject privacy and strictly comply with clinical ethics guidelines, the associated multi-posture synchronized mmWave radar and ECG dataset is currently undergoing rigorous privacy desensitization and anonymization screening. **Codes and subset data will be released as soon as the anonymization screening process is cleared**. Please stay tuned for future updates to this repository.
