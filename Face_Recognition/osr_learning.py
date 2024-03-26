from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def spl_training(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the single pseudo label (SPL) approach
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and return values.
    Introduce additional helper functions if desired.

    Parameters
    ----------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.

    Returns
    -------
    y_pred :    array, shape (n_samples,). The predicted class labels.
    y_score :   array, shape (n_samples,).
                The similarities or confidence scores of the predicted class labels. We assume that the scores are
                confidence/similarity values, i.e., a high value indicates that the class prediction is trustworthy.
                To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf) means high confidence.
                Please ensure that your score is formatted accordingly.
    """
    y_spl = np.where(y >= 0, 1, -1)

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(x, y_spl, test_size=0.2, random_state=42)

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=100,max_depth=20, max_features='sqrt', random_state=42)  # Consider hyperparameter tuning
    clf.fit(X_train, y_train)

    # Get predictions and prediction scores
    y_pred = clf.predict(x)
    y_scores = clf.predict_proba(x)
    y_pred_val = clf.predict(X_val)

    print("SPL Accuracy on all training data:", accuracy_score(y_spl, y_pred))
    print("SPL Accuracy on test data (test_size=0.2):", accuracy_score(y_val, y_pred_val))
    #sns.heatmap(confusion_matrix(y_val, y_pred), annot=True)
    #plt.show()

    return y_pred, y_scores

def mlp_training(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and return values.
    Introduce additional helper functions if desired.

    Parameters
    ----------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.

    Returns
    -------
    y_pred :    array, shape (n_samples,). The predicted class labels.
    y_score :   array, shape (n_samples,).
                The similarities or confidence scores of the predicted class labels. We assume that the scores are
                confidence/similarity values, i.e., a high value indicates that the class prediction is trustworthy.
                To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf) means high confidence.
                Please ensure that your score is formatted accordingly.
    """
    # Assign a unique label to each KUC
    max_kc_label = int(max(y[y >= 0]))

    # Assign a unique label to each KUC
    unique_kuc_labels = range(max_kc_label + 1, max_kc_label + 1 + len(set(y[y < 0])))
    kuc_mapping = {kuc_label: new_label for kuc_label, new_label in zip(set(y[y < 0]), unique_kuc_labels)}
    y_mpl = np.array([kuc_mapping[label] if label in kuc_mapping else label for label in y])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(x, y_mpl, test_size=0.2, random_state=42)

    # Classifier
    clf = SVC(C=10,class_weight='balanced',coef0=0.0,decision_function_shape='ovo',degree=3,gamma='scale', kernel='linear',tol=0.001,random_state=42,probability=True)
    clf.fit(X_train, y_train)

    # Get predictions and prediction scores
    y_pred = clf.predict(x)
    y_scores = clf.predict_proba(x)
    y_pred_val = clf.predict(X_val)

    # MPL Analysis
    print("MPL Accuracy on all training data:", accuracy_score(y_mpl, y_pred))
    print("MPL Accuracy on test data (test_size=0.2):", accuracy_score(y_val, y_pred_val))
    #print(classification_report(y_val, y_pred))
    #sns.heatmap(confusion_matrix(y_val, y_pred), annot=True)
    #plt.show()

    return y_pred, y_scores


def load_challenge_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge validation data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    path_to_challenge_validation_data = "/Users/melihekinci/Documents/FAU_Courses/ThirdSemester/Computer Vision/Week 4/exercise04_prem_melih/challenge_validation_data.csv"
    df = pd.read_csv(path_to_challenge_validation_data, header=None)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return x, y


if __name__ == '__main__':
    _x, _y = load_challenge_validation_data()
  
    y_pred_spl, y_score_spl = spl_training(_x, _y)

    y_pred_mpl, y_score_mpl= mlp_training(_x, _y)
