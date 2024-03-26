import os
import shlex
import argparse
from tqdm import tqdm

from sklearn.linear_model import Ridge
import numpy as np
import gzip
import _pickle as cPickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
import multiprocessing
from multiprocessing import Pool
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
import numpy as np
import functools

from multiprocessing import Pool, cpu_count



def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def getFiles_v2(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.desc', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors


def compute_and_save_desc(file_path, new_directory):
    # Compute descriptors
    descs = computeDescs(file_path)

    # Save descriptors to a file in the new directory
    desc_filename = os.path.splitext(os.path.basename(file_path))[0] + '.desc'
    desc_file_path = os.path.join(new_directory, desc_filename)
    with gzip.open(desc_file_path, 'wb') as fOut:
        cPickle.dump(descs, fOut)


def loadRandomDescriptors_v2(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')

        # Ensure desc is a numpy array
        desc = np.array(desc)

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def loadRandomDescriptors_v3(all_files, max_descriptors, seed=None):
    """
    Load roughly `max_descriptors` random descriptors from a random subset of files.
    Parameters:
        all_files: list of all filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
        seed: random seed for reproducibility (optional)
    Returns: QxD matrix of descriptors
    """
    np.random.seed(seed)  # Set the random seed if provided
    selected_files = np.random.choice(all_files, size=min(100, len(all_files)), replace=False)

    max_descs_per_file = int(max_descriptors / len(selected_files))
    descriptors = []

    for f in tqdm(selected_files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        
        desc = np.array(desc)
        indices = np.random.choice(len(desc), min(len(desc), max_descs_per_file), replace=False)
        desc = desc[indices]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors



def computeDescs(filename):
    # Load the image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints
    keypoints = sift.detect(img, None)

    # Modify keypoints: set all angles to 0
    for kp in keypoints:
        kp.angle = 0

    # Compute SIFT descriptors
    _, descriptors = sift.compute(img, keypoints)

    # Apply Hellinger normalization
    # Step 1: L1 normalization
    descriptors /= np.sum(descriptors, axis=1, keepdims=True)

    # Step 2: Sign square root
    descriptors = np.sqrt(np.abs(descriptors)) * np.sign(descriptors)

    return descriptors


def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    # dummy = np.array([42])
    # return dummy
    # Flatten the descriptors to be compatible with MiniBatchKMeans
    descriptors_flat = descriptors.reshape(-1, descriptors.shape[-1])

    # Use MiniBatchKMeans to compute the codebook (vocabulary)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors_flat)

    # Get the cluster centers (codebook)
    mus = kmeans.cluster_centers_

    return mus    

def generate_multiple_codebooks(descriptors, n_clusters, num_codebooks, seeds):
    codebooks = []
    for seed in seeds:
        # Flatten the descriptors for MiniBatchKMeans
        descriptors_flat = descriptors.reshape(-1, descriptors.shape[-1])

        # Use MiniBatchKMeans with different seeds
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(descriptors_flat)

        # Append the cluster centers (codebook) to the list
        codebooks.append(kmeans.cluster_centers_)

    return codebooks




    
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # Create a BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors with cluster centers using knnMatch
    matches = bf.knnMatch(descriptors, clusters, k=1)

    # Create assignment matrix
    assignment = np.zeros((len(descriptors), len(clusters)))

    for i, match in enumerate(matches):
        best_match = match[0]
        best_cluster_index = best_match.trainIdx
        assignment[i, best_cluster_index] = 1

    return assignment


def fit_ridge_parallel(residuals, cluster_indices, gamma):
    D = residuals.shape[1]
    coefs = np.zeros(D)
    for d in range(D):
        ridge = Ridge(alpha=gamma, fit_intercept=False, max_iter=500, solver='sparse_cg')
        ridge.fit(cluster_indices.reshape(-1, 1), residuals[:, d])
        coefs[d] = ridge.coef_
    return coefs

def fit_ridge_for_cluster(residuals, gamma):
    D = residuals.shape[1]
    coefs = np.zeros(D)
    for d in range(D):
        ridge = Ridge(alpha=gamma, fit_intercept=False, max_iter=100, solver='sag')  # Reduced max_iter and simpler solver
        ridge.fit(residuals[:, [d]], residuals[:, d])
        coefs[d] = ridge.coef_[0]
    return coefs

def generate_multiple_codebooks_v2(files, n_clusters, num_codebooks, max_descriptors, seeds):
    codebooks = []
    for i, seed in enumerate(seeds):
        print(f"Generating codebook {i + 1}/{num_codebooks} with seed {seed}...")
        descriptors = loadRandomDescriptors_v3(files, max_descriptors, seed)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
        kmeans.fit(descriptors)
        codebooks.append(kmeans.cluster_centers_)
    return codebooks


def vlad(files, mus, powernorm, gamma, gmp=False):
    K, D = mus.shape
    encodings = []

    for f in tqdm(files):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)

        T, D = desc.shape
        f_enc = np.zeros((D * K), dtype=np.float32)
        for k in range(K):
            cluster_indices = np.where(a[:, k] == 1)[0]

            if len(cluster_indices) > 0:
                residuals = desc[cluster_indices] - mus[k]

                if gmp:
                    #print("GMP block")
                    # Ridge regression for all dimensions of the cluster at once
                    ridge = Ridge(alpha=gamma, fit_intercept=False, max_iter=500, solver='sparse_cg')
                    ridge.fit(residuals, np.ones(len(cluster_indices)))
                    f_enc[k * D: (k + 1) * D] = ridge.coef_.flatten()
                else:
                    # Sum Pooling
                    f_enc[k * D: (k + 1) * D] = np.sum(residuals, axis=0)

        # Power normalization and L2 normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        f_enc /= np.linalg.norm(f_enc)

        encodings.append(f_enc)

    return encodings

def multi_vlad(files, codebooks, powernorm, gamma, gmp):
    all_encodings = []
    for f in files:
        file_encodings = []
        for mus in codebooks:
            vlad_encoding = vlad([f], mus, powernorm, gamma, gmp)  # Single file passed
            file_encodings.append(vlad_encoding[0])  # vlad returns a list, take the first element
        concatenated_encoding = np.concatenate(file_encodings, axis=0)  # Concatenate along the 0th axis
        all_encodings.append(concatenated_encoding)
    return np.array(all_encodings)


def apply_pca(encodings):
    pca = PCA(n_components=1000, whiten=True)
    pca.fit(encodings)
    reduced_encodings = pca.transform(encodings)
    return reduced_encodings, pca



def exemplar_classification(train_encodings, test_encodings):
    new_global_descriptors = []
    for test_encoding in test_encodings:
        # Create labels for the SVM: 0 for train encodings (negative), 1 for the test encoding (positive)
        labels = [0] * len(train_encodings) + [1]
        training_data = np.vstack((train_encodings, test_encoding))

        # Train the SVM
        svm = LinearSVC(C=1000, class_weight='balanced')
        svm.fit(training_data, labels)

        # Normalize the weight vector and add to the list of new global descriptors
        normalized_vector = normalize(svm.coef_, norm='l2')
        new_global_descriptors.append(normalized_vector.ravel())

    return new_global_descriptors

def train_svm(train_encodings, test_encoding, C_value):
    labels = [0] * len(train_encodings) + [1]
    training_data = np.vstack((train_encodings, test_encoding))
    svm = LinearSVC(C=C_value, class_weight='balanced')
    svm.fit(training_data, labels)
    normalized_vector = normalize(svm.coef_, norm='l2')
    return normalized_vector.ravel()

def exemplar_classification_parallel(train_encodings, test_encodings, C_value, n_jobs=-1):
    # Set the number of jobs to the number of CPUs if n_jobs is -1
    if n_jobs == -1:
        n_jobs = cpu_count()

    # If there is only one test encoding, no need to use multiprocessing
    if len(test_encodings) == 1 or n_jobs == 1:
        return [train_svm(train_encodings, test_encodings[0], C_value)]

    # Using multiprocessing for multiple test encodings
    with Pool(processes=n_jobs) as pool:
        new_global_descriptors = pool.map(
            functools.partial(train_svm, train_encodings, C_value=C_value),
            test_encodings
        )
    return new_global_descriptors


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    normalized_encs = normalize(encs, axis=1, norm='l2')
    dists = 1 - np.dot(normalized_encs, normalized_encs.T)

    # Mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles(args.in_train, args.suffix,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
        print('> loaded {} descriptors:'.format(len(descriptors)))

        # cluster centers
        print('> compute dictionary')
        mus = dictionary(descriptors, n_clusters=100)
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_test = vlad(files_test, mus, args.powernorm, gmp=args.gmp, gamma=args.gamma)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
   
    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        # TODO
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)

    print('> esvm computation')
    new_encs_test = esvm(enc_test, enc_train, C=args.C)

    # Save the new encodings obtained from E-SVM to a file
    fname_esvm = 'enc_test_esvm.pkl.gz'
    with gzip.open(fname_esvm, 'wb') as fOut:
        cPickle.dump(new_encs_test, fOut, -1)

    # eval
    print('> evaluate esvm')
    evaluate(new_encs_test, labels_test)
