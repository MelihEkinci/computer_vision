import numpy as np
import pickle

UNKNOWN_LABEL = -1  # Define a constant for unknown labels

class OpenSetEvaluation:

    def __init__(self, classifier, false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):
        self.false_alarm_rate_range = false_alarm_rate_range
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []
        self.classifier = classifier

    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='latin1')

    def run(self):
        self.classifier.fit(self.train_embeddings, self.train_labels)
        predicted_labels, predicted_similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        similarity_thresholds = []
        identification_rates = []

        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(predicted_similarities, far)
            thresholded_labels = [label if similarity >= threshold else UNKNOWN_LABEL for label, similarity in zip(predicted_labels, predicted_similarities)]
            identification_rate = self.calc_identification_rate(thresholded_labels, self.test_labels, threshold)

            similarity_thresholds.append(threshold)
            identification_rates.append(identification_rate)

        evaluation_results = {'similarity_thresholds': similarity_thresholds, 'identification_rates': identification_rates}
        return evaluation_results

    def select_similarity_threshold(self, similarities, false_alarm_rate):
        unknown_subject_similarities = [sim for sim, label in zip(similarities, self.test_labels) if label == UNKNOWN_LABEL]
        threshold = np.percentile(unknown_subject_similarities, 100 * (1 - false_alarm_rate))
        return threshold

    def calc_identification_rate(self, predicted_labels, actual_labels, threshold):
        # Filter for known subjects
        known_subject_indices = [i for i, label in enumerate(actual_labels) if label != UNKNOWN_LABEL]
        correct_predictions = sum(predicted_labels[i] == actual_labels[i] and predicted_labels[i] != UNKNOWN_LABEL for i in known_subject_indices)
        identification_rate = correct_predictions / len(known_subject_indices)
        return identification_rate
