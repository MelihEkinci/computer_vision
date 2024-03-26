from __future__ import division
from skimage.segmentation import felzenszwalb
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    # Apply Felzenszwalb algorithm
    segments_fz = felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    if im_orig.shape[2] == 3:  # Check if the image is RGB
        im_mask = np.dstack([im_orig, segments_fz])

    return im_mask

    #return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    return np.sum(np.minimum(r1["hist_colour"], r2["hist_colour"]))


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """

    return np.sum(np.minimum(r1["hist_texture"], r2["hist_texture"]))


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    combined_size = r1["size"] + r2["size"]
    unused_area_ratio = 1.0 - combined_size / imsize
    return unused_area_ratio


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    # Calculate bounding box coordinates
    min_x = min(r1["min_x"], r2["min_x"])
    max_x = max(r1["max_x"], r2["max_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_y = max(r1["max_y"], r2["max_y"])

    # Calculate bounding box area
    bounding_box_area = (max_x - min_x) * (max_y - min_y)

    # Calculate the fill ratio
    fill_ratio = (r1["size"] + r2["size"]) / bounding_box_area

    # Adjust the fill similarity based on the image size
    fill_similarity = fill_ratio * (bounding_box_area / imsize)
    return fill_similarity

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img,bins=25):

    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    # Initialize an empty array for the histograms
    histograms = np.array([])

    # Iterate over each color channel (HSV)
    for channel in range(3):
        # Calculate histogram for each channel
        hist, _ = np.histogram(img[:, channel], bins=BINS, range=(0, 255))
        # Concatenate the histogram to the result
        histograms = np.concatenate([histograms, hist])

    # Normalize the histogram
    histograms /= img.shape[0]  # Normalize by the number of pixels

    return histograms

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    # Number of points and radius for LBP
    numPoints = 8
    radius = 1

    # Loop through each color channel and calculate LBP
    for i in range(3):  # Assuming the image has three color channels
        lbp = skimage.feature.local_binary_pattern(img[:, :, i], numPoints, radius, method="uniform")
        ret[:, :, i] = lbp

    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    CHANNELS = 3 
    histograms = np.array([])

    for channel in range(CHANNELS):
        # Extract the channel
        channel_data = img[:, channel]

        # Calculate histogram for the channel
        hist, _ = np.histogram(channel_data, bins=BINS, range=(0.0, 1.0))

        # Concatenate the histogram to the result
        histograms = np.concatenate([histograms, hist])

    # Normalize the histogram
    histograms /= img.shape[0]  # Normalize by the number of pixels

    return histograms


def calculate_bounding_boxes(img, regions):
    for y, row in enumerate(img):
        for x, pixel in enumerate(row):
            label = pixel[3]
            if label not in regions:
                regions[label] = {
                    "min_x": np.inf, "min_y": np.inf,
                    "max_x": 0, "max_y": 0, "labels": label
                }
            regions[label]["min_x"] = min(regions[label]["min_x"], x)
            regions[label]["min_y"] = min(regions[label]["min_y"], y)
            regions[label]["max_x"] = max(regions[label]["max_x"], x)
            regions[label]["max_y"] = max(regions[label]["max_y"], y)

def calculate_histograms(img, hsv, tex_grad, regions):
    for label, props in regions.items():
        mask = img[:, :, 3] == label
        masked_pixels = img[mask]
        #print(len(masked_pixels))
        regions[label]["size"] = len(masked_pixels) // 4
        regions[label]["hist_colour"] = calc_colour_hist(masked_pixels)
        regions[label]["hist_texture"] = calc_texture_hist(tex_grad[mask])

def extract_regions(img):
    regions = {}
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    tex_grad = calc_texture_gradient(img)

    calculate_bounding_boxes(img, regions)
    calculate_histograms(img, hsv, tex_grad, regions)

    return regions

def extract_neighbours(regions):
    def are_neighbours(region_a, region_b):
        """
        Check if two regions are neighbours (intersect).
        """
        return (
            (region_a["min_x"] < region_b["min_x"] < region_a["max_x"] or region_a["min_x"] < region_b["max_x"] < region_a["max_x"]) and
            (region_a["min_y"] < region_b["min_y"] < region_a["max_y"] or region_a["min_y"] < region_b["max_y"] < region_a["max_y"])
        )

    neighbours = []
    region_items = list(regions.items())

    for i, (label_a, region_a) in enumerate(region_items[:-1]):
        for label_b, region_b in region_items[i + 1:]:
            if are_neighbours(region_a, region_b):
                neighbours.append(((label_a, region_a), (label_b, region_b)))

    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    rt={
        "min_x": min(r1["min_x"], r2["min_x"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_colour": (r1["hist_colour"] * r1["size"] + r2["hist_colour"] * r2["size"]) / new_size,
        "hist_texture": (r1["hist_texture"] * r1["size"] + r2["hist_texture"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)
    print("1.Segments are generated")

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)
    print("2.Regions are extracted")

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)
    print("3.Neighbours are extracted")
    #print(neighbours)
    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])
     
        # Task 5: Mark similarities for regions to be removed
        regions_to_remove = [k for k, v in S.items() if i in k or j in k]


        # Task 6: Remove old similarities of related regions
        for k in regions_to_remove:
            del S[k]


        # Task 7: Calculate similarities with the new region
        for k in regions_to_remove:
            if k != (i, j):
                n = k[1] if k[0] in (i, j) else k[0]
                S[(t, n)] = calc_sim(R[t], R[n], imsize)



    # Task 8: Generating the final regions from R
    print("4.Generating final regions")
    regions = [{
        'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
        'size': r['size'],
        'labels': r['labels']
    } for k, r in R.items()]



    return image, regions


