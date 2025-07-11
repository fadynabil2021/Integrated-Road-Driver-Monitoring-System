# HeadPose_and_EyeGaze_Estimation

## Overview
This directory contains code and resources for head pose and eye gaze estimation using deep learning models. The project is structured to facilitate training, evaluation, and inference on driver monitoring tasks, with a focus on automotive applications. The codebase is modular, supporting data augmentation, custom loss functions, and advanced model architectures such as attention mechanisms.

## Directory Structure

- `data/` — Data loading and augmentation utilities:
  - `dataset.py`: Dataset handling and preprocessing.
  - `augmentation.py`: Data augmentation techniques.
  - `__init__.py`
- `evaluation/` — Evaluation scripts and plotting tools:
  - `evaluate.py`: Evaluation routines for model performance.
  - `plots.py`: Visualization and plotting utilities.
  - `__init__.py`
- `models/` — Model architectures:
  - `head_gaze_model.py`: Main model definition for head pose and gaze estimation.
  - `attention.py`: Attention mechanism implementation.
  - `__init__.py`
- `training/` — Training utilities:
  - `train.py`: Training loop and logic.
  - `metrics.py`: Custom metrics for evaluation.
  - `losses.py`: Custom loss functions.
  - `__init__.py`
- `Best_Model/` — Pretrained model weights:
  - `best_model_train_val_1_2_3_4_5_6_7_8_9_10_12_13_14_17_18_test_15_19_20_mobilenet_v2.pth`: Pretrained model checkpoint.
- `main.py` — Entry point for running the pipeline.
- `pipeline.py` — Pipeline orchestration script.
- `__init__.py`

## Dataset: AutoPOSE
This project leverages the AutoPOSE dataset, a large-scale automotive driver head pose and gaze dataset. AutoPOSE provides high-quality, annotated images of drivers in real vehicles, enabling robust training and evaluation of head pose and gaze estimation models. The dataset includes a deep head orientation baseline and is designed to support research in driver monitoring and safety systems.

**Citation:**
```
@INPROCEEDINGS{Selim2020AutoPOSE,
  author = {Mohamed Selim and Ahmet Firintepe and Alain Pagani and Didier Stricker},
  title = {AutoPOSE: Large-Scale Automotive Driver Head Pose and Gaze Dataset with Deep Head Orientation Baseline},
  booktitle = {International Conference on Computer Vision Theory and Applications (VISAPP)},
  year = {2020},
  url = {http://autopose.dfki.de}
}
```

For more information about the dataset, visit [AutoPOSE Dataset Website](http://autopose.dfki.de).

## Usage
1. Place your data in the appropriate directory and update configuration as needed.
2. Use `main.py` or `pipeline.py` to train or evaluate models.
3. Pretrained weights are available in the `Best_Model/` directory.

## Requirements
- Python 3.x
- TensorFlow, PyTorch, and other dependencies (see your environment setup)

## Acknowledgements
If you use this code or the AutoPOSE dataset, please cite the original paper as shown above. 