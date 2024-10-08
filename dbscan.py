import cv2
import numpy as np
from glob import glob
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0


def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0


def calculate_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area if hull_area != 0 else 0


def calculate_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    entropy = shannon_entropy(gray)

    # Compute GLCM and its properties
    glcm = graycomatrix(gray, distances=[1], angles=[
                        0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return lbp.mean(), entropy, contrast, energy, homogeneity, correlation


# Load random image from dataset
image_list = glob(os.path.join('dataset', '**/*.jpg'), recursive=True)
image_path = random.choice(image_list)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract features (roundness, aspect ratio, solidity, texture features)
features = []
for contour in contours:
    roundness = calculate_roundness(contour)
    solidity = calculate_solidity(contour)
    aspect_ratio = calculate_aspect_ratio(contour)
    lbp, entropy, contrast, energy, homogeneity, correlation = calculate_texture_features(
        image)
    if roundness is not None and solidity is not None:
        features.append([roundness, solidity, lbp, entropy,
                        contrast, energy, homogeneity, correlation, aspect_ratio])

# Check if we have enough samples
if len(features) >= 2:
    features_array = np.array(features)

    # Standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # Apply DBSCAN for clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # You can tune eps and min_samples
    dbscan.fit(features_scaled)
    labels = dbscan.labels_

    # Filter out noise (-1 labels)
    if len(np.unique(labels)) > 1:
        # Train SVM classifier on the clustered data
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(features_scaled, labels)

        # Dimensionality Reduction with PCA
        pca = PCA(n_components='mle')
        features_pca = pca.fit_transform(features_scaled)

        # Plot PCA reduced features
        plt.scatter(features_pca[:, 0],
                    features_pca[:, 1], c=labels, cmap='viridis')
        plt.title('PCA of Spore Features (DBSCAN Clustering)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar()
        plt.show()

        # Classify each spore using the SVM
        spore_types = []
        for feature in features_scaled:
            spore_type = 'Monolete' if svm_classifier.predict(
                [feature]) == 0 else 'Trilete'
            spore_types.append(spore_type)

        # Print the classification results
        for i, spore_type in enumerate(spore_types):
            print(f'Spore {i}: {spore_type}')

        # Annotate image with classification (example: using dominant classification)
        x, y = 10, 30  # Position for the label text
        dominant_spore_type = 'Monolete' if labels[0] == 0 else 'Trilete'
        cv2.putText(image, f'Spore Type: {dominant_spore_type}',
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Save or display the annotated image
        cv2.imshow('Spore Classification', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save the image
        cv2.imwrite('classified_spores_dbscan.jpg', image)

    else:
        print('DBSCAN found only noise or insufficient clusters.')

else:
    print('Not enough samples to perform clustering.')
