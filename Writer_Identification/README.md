# Computer Vision Project: Writer Identification with VLAD Encoding

## Overview

This repository houses the implementation and findings of our computer vision project focused on Writer Identification using the ICDAR17 Historical Writer Identification (WI) Dataset. We employ the Bag of Visual Words model alongside VLAD encoding, power normalization, and SVM-based exemplar classification to achieve this. This approach was inspired by the work of Florian Kordon, Mathias Seuret, Vincent Christlein, Thomas Kohler, and Mathias Zinnen and their proposed exercises.

### Project Components

- **Codebook Generation**: Utilizes MiniBatchKMeans for creating a visual dictionary.
- **VLAD Encoding & Normalization**: Encodes local descriptors with respect to a visual dictionary and applies power normalization to account for visual burstiness.
- **Exemplar Classification**: Employs LinearSVC for individual SVM creation for each global representation in the test set.
- **Feature Extraction & Generalized Max Pooling (Bonus Tasks)**: Compares pre-computed descriptors with traditional SIFT descriptors and implements generalized max pooling for improved encoding.
- **PCA Whitening and Multi-VLAD (Bonus Task)**: Enhances results by addressing the randomness of k-means through multiple codebooks and PCA whitening.

## Repository Structure

- `exercise3_withbonus.ipynb`: Jupyter notebook detailing the implementation and analysis of the writer identification system.
- `skeleton_with_bonus.py`: Python script that includes the base code structure and additional bonus task implementations.


## Usage

To run the writer identification system:

```bash
python skeleton_with_bonus.py <path_to_dataset> <path_to_output>
```

## Dependencies

- Python 3.x
- NumPy
- SciKit-Learn
- OpenCV (Optional for SIFT features)
- Other dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Contributions

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.


## Acknowledgments

Special thanks to the course instructors Vincent Christlein for their guidance and the exercises that inspired this project.
