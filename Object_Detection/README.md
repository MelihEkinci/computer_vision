# Selective Search and Object Detection in Computer Vision

## Project Overview

This repository contains the implementation and exploration of selective search and object detection algorithms as part of the Computer Vision project for the Winter Term 2023/24, instructed by Mathias Zinnen. The project is structured into two main exercises:
- **Exercise 5.1**: Implementation of the selective search algorithm for object detection, mandatory for all course participants.
- **Exercise 5.2**: Development of a detection pipeline leveraging region proposals for object detection, mandatory for 10 ECTS participants.

## Installation

Before running any code, ensure you have a compatible Python environment. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes essential libraries such as `scikit-image`, `numpy`, and `matplotlib`.

## Repository Structure

- `detection_pipeline.ipynb`: Jupyter notebook with the implementation details and step-by-step explanation of the selective search and detection pipeline.
- `detection_pipeline.py`: Python script for the detection pipeline, coordinating the selective search and subsequent object detection.
- `main2.py`: Entry point for running selective search experiments and generating region proposals.
- `selective_search.py`: Core implementation of the selective search algorithm.
- `requirements.txt`: Lists all the necessary Python packages to run the project.

## Usage Instructions

### Running Selective Search

To execute the selective search on the provided dataset and generate region proposals:

```bash
python main2.py /path/to/image
```

This script processes images from the specified data directories and saves the output in the results folder.

### Object Detection Pipeline

For those completing the 10 ECTS version of the course, follow the steps outlined in `detection_pipeline.ipynb` to train and test your object detection model using the generated region proposals.

### Visualization and Analysis

Open and run the `detection_pipeline.ipynb` notebook for a detailed walkthrough of the detection pipeline, including visualizations and analysis of the results:

```bash
jupyter notebook detection_pipeline.ipynb
```

## Contributions

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.


## Acknowledgments

Special thanks to Mathias Zinnen for providing the framework and guidance for this computer vision project, and to all contributors to the datasets and tools used.
