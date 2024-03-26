# Basic Face Recognition System

## Overview

This repository contains the implementation of a basic face recognition system developed as part of the Computer Vision Project for the Winter Term 2023/24, under the guidance of Florian Kordon, Vincent Christlein, Mathias Seuret, Thomas Köhler, and Mathias Zinnen. The project aims to train a system on facial data for identification and re-identification purposes in video data, covering both supervised and unsupervised learning techniques.

### System Components

- **Training Module (`training.py`)**: Supports identification mode for collecting labeled training data and a clustering mode for acquiring unlabeled data.
- **Testing Module (`test.py`)**: Utilizes trained models for face identification or re-identification and visualizes results.

## Repository Contents

- `osr_learning.py`: Implementation of open-set recognition with known classes and known unknown classes.
- `ex4_additional.ipynb`: Jupyter notebook with additional analysis and results.
- `evaluation_final.py`: Script for evaluating face recognition performance.
- `dir_curve.py`: Script for generating Detection and Identification Rate (DIR) curves.
- `face_recognition_final.py`: Final implementation of face recognition functionalities.
- `face_detector_final.py`: Face detection, tracking, and alignment module.
- `training.py`, `test.py`: Main training and testing scripts for the face recognition system.
- `requirements.txt`: List of Python package dependencies for the project.

## Installation

Ensure you have Python 3.x installed, then clone this repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage

- To train the system, run:
  ```bash
  python training.py
  ```

- To test the system, run:
  ```bash
  python test.py
  ```

For detailed usage of other scripts, refer to the inline documentation within each file.

## Dependencies

- mtcnn>=0.1.0
- numpy>=1.16.4
- opencv-python>=4.1
- scipy>=1.3.0

## Contributions

Contributions to this project are welcome. Please open an issue to discuss proposed changes or notify of any issues.


## Acknowledgments

Special thanks to  Thomas Köhler for his invaluable guidance and to all contributors to the YouTube Faces database and other resources utilized in this project.
