import cv2
import numpy as np
from glob import glob
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import mahotas
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h


def calculate_roundness(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    roundness = (4 * np.pi * area) / (perimeter ** 2)
    return roundness


def calculate_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return None
    return float(area) / hull_area


def calculate_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    entropy = shannon_entropy(gray)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return lbp.mean(), entropy, haralick.mean()


def classify_spore(features_scaled, kmeans_model):
    cluster = kmeans_model.predict(features_scaled)
    return 'Monolete' if cluster == 0 else 'Trilete'


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
    aspect_ratio = calculate_aspect_ratio(contour)
    solidity = calculate_solidity(contour)
    lbp, entropy, haralick = calculate_texture_features(image)
    if roundness is not None and aspect_ratio is not None and solidity is not None:
        features.append(
            [roundness, aspect_ratio, solidity, lbp, entropy, haralick])

# Check if we have enough samples
if len(features) >= 2:
    features_array = np.array(features)

    # Standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # Apply GridSearchCV to tune KMeans
    param_grid = {'n_clusters': [2, 3, 4], 'init': [
        'k-means++', 'random'], 'n_init': [10, 15, 20]}
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
    grid_search = GridSearchCV(kmeans, param_grid, cv=4)
    grid_search.fit(features_scaled)

    # Best KMeans model after tuning
    kmeans_best = grid_search.best_estimator_

    # Predict cluster labels
    labels = kmeans_best.labels_

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    # Plot PCA reduced features
    plt.scatter(features_pca[:, 0], features_pca[:, 1],
                c=labels, cmap='viridis')
    plt.title('PCA of Spore Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

    # Classify each spore
    spore_types = []
    for feature in features_scaled:
        spore_type = classify_spore([feature], kmeans_best)
        spore_types.append(spore_type)

    # Print the classification results
    for i, spore_type in enumerate(spore_types):
        print(f'Spore {i}: {spore_type}')

    # Annotate image with classification (example: using dominant classification)
    x, y = 10, 30  # Position for the label text
    dominant_spore_type = 'Monolete' if labels[0] == 0 else 'Trilete'
    cv2.putText(image, f'Spore Type: {dominant_spore_type}', (
        x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Save or display the annotated image
    cv2.imshow('Spore Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the image
    cv2.imwrite('classified_spores.jpg', image)

else:
    print('Not enough samples to perform clustering.')
