import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from glob import glob
import random


def extract_features(image):
    features = []
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresholded, 100, 200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        circularity = calculate_circularity(contour)
        ridge_count = detect_ridges(image[y:y+h, x:x+w])

        features.append([aspect_ratio, circularity, ridge_count])
    return np.array(features), contours


def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / (perimeter * perimeter)


def detect_ridges(spore_roi):
    edges_roi = cv2.Canny(spore_roi, 100, 200)
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180,
                            threshold=20, minLineLength=10, maxLineGap=5)
    if lines is not None:
        return len(lines)
    return 0


# Load and process the image
image_list = glob(os.path.join('dataset', '**/*.jpg'), recursive=True)
image_path = random.choice(image_list)
print(image_path)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
features, contours = extract_features(image)

# Ensure there are features to process
if features.size == 0:
    raise ValueError(
        "No features extracted from the image. Ensure the image contains detectable objects.")

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_features)

# Get cluster labels
labels = kmeans.labels_

cluster_1_features = scaled_features[labels == 0]
cluster_2_features = scaled_features[labels == 1]

avg_aspect_ratio_cluster_1 = np.mean(cluster_1_features[:, 0])
avg_aspect_ratio_cluster_2 = np.mean(cluster_2_features[:, 0])

if avg_aspect_ratio_cluster_1 > avg_aspect_ratio_cluster_2:
    cluster_1_label = 'Monolete'
    cluster_2_label = 'Trilete'
else:
    cluster_1_label = 'Trilete'
    cluster_2_label = 'Monolete'

# Check for matching contours and labels
if len(contours) != len(labels):
    print(
        f"Warning: The number of contours ({len(contours)}) does not match the number of labels ({len(labels)}).")

# Draw bounding boxes and labels on the image
image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw bounding boxes and labels for contours with features
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 100:
        continue

    # Ensure these variables are defined
    x, y, w, h = cv2.boundingRect(contour)
    if i >= len(labels):
        continue

    # Extract feature for this contour
    spore_roi = image[y:y+h, x:x+w]
    contour_feature = extract_features(spore_roi)

    if len(contour_feature) == 0:
        continue

    scaled_contour_feature = scaler.transform(contour_feature)
    contour_label = kmeans.predict(scaled_contour_feature)[0]

    # Ensure that index is within bounds
    if i >= len(labels):
        continue

    label = cluster_1_label if contour_label == 0 else cluster_2_label
    color = (0, 255, 0) if label == 'Trilete' else (
        255, 0, 0)  # Green for Trilete, Blue for Monolete

    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image_bgr, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display the labeled image
cv2.imshow('Labeled Image', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
