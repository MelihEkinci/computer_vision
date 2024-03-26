import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from random import sample

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")

    # Predict embedding from a given face image.
    def predict(self, face):
        print("Original face shape:", face.shape)
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings
    
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=11, max_distance=0.8, min_prob=0.5):
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = {}

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    def update(self, face, label):
        # Extract the embedding using the FaceNet model
        embedding = self.facenet.predict(face)
        print("Embedding shape:", embedding.shape)  # This should print (128,)

        # If the label does not exist in the gallery, add it with the embedding
        if label not in self.labels:
            self.labels.append(label)
            self.embeddings[label] = [embedding]
        else:
            # If the label exists, append the new embedding to the list of embeddings for that label
            self.embeddings[label].append(embedding)

    def predict(self, face, k=5, distance_threshold=0.8, probability_threshold=0.5):
        # Extract the embedding for the given face
        face_embedding = self.facenet.predict(face)
        #print(face.shape)
        #print(face_embedding.shape)

        # Flatten the embeddings into a 2D array for k-NN
        all_embeddings = []
        all_labels = []
        
        for label, embeddings_list in self.embeddings.items():
            for embedding in embeddings_list:
                print("Embedding shape in flatten process:", embedding.shape) 
                all_embeddings.append(embedding)
                all_labels.append(label)


        
        all_embeddings = np.array(all_embeddings)  # Ensure this is a 2D array
        print("All embeddings shape:", all_embeddings.shape)

        # Use NearestNeighbors from sklearn to find the k nearest neighbors
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(all_embeddings)
        distances, indices = knn.kneighbors([face_embedding])

        # Get the labels of the k nearest neighbors
        neighbor_labels = [all_labels[idx] for idx in indices[0]]

        # Count the occurrences of each label in the k nearest neighbors
        label_counts = {label: neighbor_labels.count(label) for label in set(neighbor_labels)}

        # Determine the majority label and its count
        majority_label, majority_count = max(label_counts.items(), key=lambda x: x[1])

        # Calculate the posterior probability
        posterior_probability = majority_count / k

        # Calculate the minimum distance to the majority class
        majority_class_distances = distances[0][np.array(neighbor_labels) == majority_label]
        min_distance_to_class = np.min(majority_class_distances)

        # Open-set protocol decision rule
        if min_distance_to_class > distance_threshold or posterior_probability < probability_threshold:
            return "unknown", posterior_probability, min_distance_to_class
        else:
            return majority_label, posterior_probability, min_distance_to_class

# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self,num_clusters=2, max_iter=100):
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    def update(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.append(self.embeddings, [embedding], axis=0)
        
    def fit(self):
        # Randomly choose k embeddings as initial cluster centers using NumPy
        indices = np.random.choice(len(self.embeddings), self.num_clusters, replace=False)
        centers = self.embeddings[indices]
        labels = np.zeros(len(self.embeddings))

        for iteration in range(self.max_iter):
            # Assign labels based on closest center
            for i, embedding in enumerate(self.embeddings):
                distances = [np.linalg.norm(embedding - center) for center in centers]
                labels[i] = np.argmin(distances)

            # Update centers
            for j in range(self.num_clusters):  # Use self.num_clusters instead of k
                points = self.embeddings[labels == j]
                if len(points) > 0:
                    centers[j] = np.mean(points, axis=0)

        self.centers = centers
        self.labels = labels


    def predict(self, face):
        embedding = self.facenet.predict(face)
        distances = [np.linalg.norm(embedding - center) for center in self.cluster_center]
        best_cluster = np.argmin(distances)
        return best_cluster, distances
