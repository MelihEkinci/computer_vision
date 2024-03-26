from __future__ import (
    division,
    print_function,
)
import sys
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from skimage.segmentation import felzenszwalb
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import pickle
import os
import json
import selective_search
from selective_search import selective_search

def convert_numpy_to_python(data):
    """
    Convert numpy data types to native Python types for JSON serialization.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()  # Replacing np.asscalar with .item()
    return data

def save_proposals(image_path, output_path, selective_search_func):
    """
    Apply selective search on an image and save the proposals.
    """
    # Load the image
    image = skimage.io.imread(image_path)

    # Generate region proposals
    _, proposals = selective_search_func(image)

    # Filter out proposals with 0 width or height
    valid_proposals = [p for p in proposals if p['rect'][2] > 0 and p['rect'][3] > 0]

    # Convert NumPy types to native Python types for JSON serialization
    for proposal in valid_proposals:
        proposal['rect'] = convert_numpy_to_python(proposal['rect'])
        proposal['size'] = convert_numpy_to_python(proposal['size'])
        proposal['labels'] = convert_numpy_to_python(proposal['labels'])

    # Save the proposals to a file
    with open(output_path, 'w') as file:
        json.dump(valid_proposals, file)

        
def process_dataset(dataset_path, selective_search_func):
    """
    Process all images in the dataset using selective search and save the proposals.
    """
    for split in ['train', 'valid']:
        split_path = os.path.join(dataset_path, split)
        output_split_path = os.path.join(dataset_path, split + '_proposals')

        # Create directory for proposals if it doesn't exist
        if not os.path.exists(output_split_path):
            os.makedirs(output_split_path)

        for image_file in os.listdir(split_path):
            if image_file.endswith('.json') or image_file.endswith('.DS_Store'):  # Skip the annotation file
                continue

            image_path = os.path.join(split_path, image_file)
            output_path = os.path.join(output_split_path, image_file + '.json')

            save_proposals(image_path, output_path, selective_search_func)


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def classify_proposals(proposals_dir, annotations, tp=0.50, tn=0.15):
    positive_samples_dict, negative_samples_dict = [], []

    for proposal_file in os.listdir(proposals_dir):
        print(proposal_file)
        if proposal_file.endswith('.json'):
            proposals = load_json(os.path.join(proposals_dir, proposal_file))
            image_base_name = os.path.splitext(proposal_file)[0]
            # Correct way to find the image_id
            image_id = next((image['id'] for image in annotations['images'] if image['file_name'].startswith(image_base_name)), None)

            if image_id is not None:
                image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
                for proposal in proposals:
                    #print(proposal)
                    proposal_box = [proposal['rect'][0], proposal['rect'][1], proposal['rect'][0] + proposal['rect'][2], proposal['rect'][1] + proposal['rect'][3]]
                    max_iou = 0

                    for ann in image_annotations:
                        ann_box = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
                        iou = calculate_iou(proposal_box, ann_box)
                        max_iou = max(max_iou, iou)

                    if max_iou >= tp:
                        positive_samples_dict.append({'proposal': proposal, 'image_name': image_base_name})
                    elif max_iou <= tn:
                        negative_samples_dict.append({'proposal': proposal, 'image_name': image_base_name})

    return positive_samples_dict, negative_samples_dict

def extract_features(image_folder, proposals):
    features = []
    i=0
    for element in proposals:
        i=i+1
        print(i+1)
        proposal=element['proposal']
        #print(proposal)
        x, y, w, h = proposal['rect']
        image_name=element['image_name']
        #print(image_name)
        image_path=image_folder+image_name
        #print(image_path)
        image = cv2.imread(image_path)
        #print(image.shape)
        #print(y,y+h,x,x+w)
        crop = image[y:y+h, x:x+w]
        #print(crop.shape)
        crop = cv2.resize(crop, (224, 224))
        crop = np.expand_dims(crop, axis=0)
        crop = preprocess_input(crop)

        feature = model.predict(crop)
        features.append(feature.flatten())
    
    return features

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features_for_script(image_path, proposals):
    features = []
    for proposal in proposals:
        #print(proposal)
        x, y, w, h = proposal['rect']
        #print(image_name)
        image_path=image_path
        #print(image_path)
        image = cv2.imread(image_path)
        #print(image.shape)
        #print(y,y+h,x,x+w)
        crop = image[y:y+h, x:x+w]
        #print(crop.shape)
        crop = cv2.resize(crop, (224, 224))
        crop = np.expand_dims(crop, axis=0)
        crop = preprocess_input(crop)

        feature = model.predict(crop)
        features.append(feature.flatten())
    features = np.array(features)
    return features


