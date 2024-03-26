import cv2
import joblib
from selective_search import selective_search  # Ensure this is correctly imported
import detection_pipeline
from detection_pipeline import extract_features,extract_features_for_script
import argparse
import numpy as np

def detect_objects(image_path, clf):
    # Load the image
    image = cv2.imread(image_path)

    # Generate proposals using selective search
    _,proposals = selective_search(image)

    valid_proposals = [p for p in proposals if p['rect'][2] > 0 and p['rect'][3] > 0]

    # Extract features for each proposal
    features = [extract_features_for_script(image_path,valid_proposals)]
    print(features)
    features = np.vstack(features)
    # Classify each proposal
    predictions = clf.predict(features)
    print(predictions)

    # Draw bounding boxes around positive predictions
    for proposal, prediction in zip(valid_proposals, predictions):
        if prediction == 1:  # Positive prediction
            x, y, w, h = proposal['rect']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Object Detection Script")
    parser.add_argument("image_path", help="Path to the image file")

    # Parse the arguments
    args = parser.parse_args()

    # Load the SVM model
    clf = joblib.load('code/svm_classifier.pkl')

    # Run object detection
    detect_objects(args.image_path, clf)

if __name__ == "__main__":
    main()
