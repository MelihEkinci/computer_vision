# Box Detection using Kinect Data

This repository contains a solution for detecting and measuring the dimensions of a box using Kinect depth data. The algorithm takes a point cloud and an amplitude image as input and estimates the length, width, and height of the box.

## Problem Description

The goal of this project is to develop an algorithm that can estimate the size of a box from a distance image captured by a Kinect sensor. The input data consists of an amplitude image, a distance image, and a point cloud, all of which are registered with each other.

The main steps involved in the algorithm are:
1. Load and preprocess the input data.
2. Apply RANSAC to find the dominant planes corresponding to the floor and the top of the box.
3. Filter the mask images using morphological operations to improve the quality of the floor and box masks.
4. Extract the largest connected component from the box mask.
5. Detect the corners of the box based on the largest connected component.
6. Calculate the length, width, and height of the box using the detected corners and the distance between the floor and box planes.

## Solution

The solution is implemented in Python using Jupyter Notebook. The main steps of the algorithm are as follows:

1. Load the input data from the provided .mat files using `scipy.io.loadmat()`.
2. Apply RANSAC to find the floor plane and create a floor mask.
3. Filter the floor mask using morphological operations to remove noise and improve the mask quality.
4. Apply RANSAC to the non-floor points to find the box plane and create a box mask.
5. Extract the largest connected component from the box mask using `scipy.ndimage.label()`.
6. Detect the corners of the box using contour detection and polygon approximation.
7. Calculate the length and width of the box by computing the distances between the detected corners.
8. Calculate the height of the box by computing the distance between the floor and box planes.
9. Visualize the results by displaying the floor mask, box mask, and detected corners.

## Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- OpenCV

## Usage

1.Clone the repository:
git clone https://github.com/your-username/computer_vision.git
2.Navigate to the repository directory:
cd Box_Detection
3. Place the input .mat files in the `data/` folder.

4. Open the Jupyter Notebook `solution.ipynb` and run the cells to execute the box detection algorithm.

5. The results will be displayed inline in the notebook, showing the detected floor mask, box mask, corners, and the estimated dimensions of the box.

## Results

The algorithm was tested on four example datasets provided in the `data/` folder. The estimated box dimensions for each example are as follows:

- Example 1: Length = 238.36, Width = 148.56, Height = 0.74
- Example 2: Length = 176.84, Width = 123.74, Height = 0.98
- Example 3: Length = 176.84, Width = 123.74, Height = 1.06
- Example 4: Length = 176.84, Width = 123.74, Height = 0.92

The corresponding visualizations for each example can be found in the `images/` folder.

## Future Work

Some potential improvements and future work for this project include:

- Optimizing the algorithm for better performance and robustness.
- Handling cases with multiple boxes or irregular box shapes.
- Incorporating additional features or constraints to improve the accuracy of box detection and dimension estimation.
- Extending the algorithm to work with other types of depth sensors or point cloud data.

## Acknowledgments

- The example datasets used in this project were provided as part of the Computer Vision course at FAU.
- The RANSAC implementation is based on the pseudocode provided in the lecture slides.
