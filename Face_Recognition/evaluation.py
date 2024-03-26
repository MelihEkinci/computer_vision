import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f,encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f,encoding='latin1')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        # Fit the classifier on the training data
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # Predict labels and similarities on the test data
        predicted_labels, predicted_similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        similarity_thresholds = []
        identification_rates = []

        # Iterate over the false alarm rates
        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(predicted_similarities, far)

            # Apply the threshold to determine predicted labels (use UNKNOWN_LABEL for those below the threshold)
            thresholded_labels = np.where(predicted_similarities >= threshold, predicted_labels, UNKNOWN_LABEL)

            # Calculate the identification rate
            id_rate = self.calc_identification_rate(thresholded_labels)

            similarity_thresholds.append(threshold)
            identification_rates.append(id_rate)

        # Report all performance measures
        evaluation_results = {
            'similarity_thresholds': similarity_thresholds,
            'identification_rates': identification_rates
        }

        return evaluation_results


# This function would be a part of the OpenSetEvaluation class.
# To integrate this function, you need to replace the existing run method with this one.


    def select_similarity_threshold(self, similarities, desired_false_alarm_rate):
        # Calculate the percentile corresponding to the desired false alarm rate.
        # This is based on the assumption that higher similarity scores are better.
        # Thus, we want the threshold at which (100 - p)% of the scores are higher.
        percentile = 100 * (1 - desired_false_alarm_rate)

        # Determine the similarity threshold
        similarity_threshold = np.percentile(similarities, percentile)

        return similarity_threshold

    def calc_identification_rate(self, prediction_labels):
        # Count the number of correct predictions
        correct_predictions = np.sum(prediction_labels == self.test_labels)

        # Calculate the identification rate
        identification_rate = correct_predictions / len(self.test_labels)

        return identification_rate
