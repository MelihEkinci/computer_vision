import cv2
from mtcnn import MTCNN
import numpy as np


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=20, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

	# ToDo: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size

    # ToDo: Track a face in a new image using template matching.
    def track_face(self, image):
        # If no reference, use detect_face to initialize tracking
        if self.reference is None:
            detection = self.detect_face(image)
            if detection is None:
                return None
            self.reference = detection['rect']
            self.template = self.crop_face(image, self.reference)  # Save the template (reference face)
            return detection

        # Define the search window in the new frame
        top, left, bottom, right = self.define_search_window(self.reference, image)
        search_window = image[top:bottom, left:right]

        # Perform template matching
        res = cv2.matchTemplate(search_window, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # Check if the tracking is reliable
        if max_val < self.tm_threshold:
            # Reinitialize tracking
            detection = self.detect_face(image)
            if detection is None:
                return None
            self.reference = detection['rect']
            self.template = self.crop_face(image, self.reference)  # Update the template
            return detection

        # Update the reference position
        top, left = top + max_loc[1], left + max_loc[0]
        self.reference = [left, top, self.reference[2], self.reference[3]]

        # Return updated tracking information
        aligned = self.align_face(image, self.reference)
        return {"rect": self.reference, "image": image, "aligned": aligned, "response": max_val}
    
    def define_search_window(self, ref_rect, image):
    # Calculate the search window boundaries
        top = max(ref_rect[1] - self.tm_window_size, 0)
        left = max(ref_rect[0] - self.tm_window_size, 0)
        bottom = min(ref_rect[1] + ref_rect[3] + self.tm_window_size, image.shape[0])
        right = min(ref_rect[0] + ref_rect[2] + self.tm_window_size, image.shape[1])
        return top, left, bottom, right

    def is_search_window_valid(self, search_window, template):
        # Check if the search window is valid
        return search_window.shape[0] >= template.shape[0] and search_window.shape[1] >= template.shape[1]

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        detections = self.detector.detect_faces(image)
        if not detections:
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(self.crop_face(image, face_rect), dsize=(self.aligned_image_size, self.aligned_image_size))

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]

