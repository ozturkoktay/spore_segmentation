import numpy as np
import cv2
from glob import glob
import os
import random


def find_polar_axis(contour):
    hull = cv2.convexHull(contour)
    max_distance = 0
    polar_axis = None
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            pt1, pt2 = hull[i][0], hull[j][0]
            distance = np.linalg.norm(pt1 - pt2)
            if distance > max_distance:
                max_distance = distance
                polar_axis = (tuple(pt1), tuple(pt2))
    return polar_axis, max_distance


def calculate_equatorial_axes(contour, polar_axis, num_axes=3):
    pt1, pt2 = np.array(polar_axis[0]), np.array(polar_axis[1])
    polar_vector = pt2 - pt1
    polar_length = np.linalg.norm(polar_vector)
    unit_polar_vector = polar_vector / polar_length
    equatorial_points = [pt1 + i * (polar_length / (num_axes + 1))
                         * unit_polar_vector for i in range(1, num_axes + 1)]
    equatorial_axes = []
    for pt in equatorial_points:
        perpendicular_vector = np.array(
            [-unit_polar_vector[1], unit_polar_vector[0]])
        equatorial_axes.append((pt, perpendicular_vector))
    return equatorial_axes


def measure_widths(image, contour, equatorial_axes):
    widths = []
    for center, axis_vector in equatorial_axes:
        extended_axis_pos = center + 1000 * axis_vector
        extended_axis_neg = center - 1000 * axis_vector
        mask = np.zeros_like(image)
        cv2.line(mask, tuple(extended_axis_neg.astype(int)),
                 tuple(extended_axis_pos.astype(int)), 255, 1)

        intersections = cv2.bitwise_and(mask, np.uint8(contour))

        intersection_points = np.transpose(np.nonzero(intersections))

        if len(intersection_points) >= 2:
            width = np.linalg.norm(
                intersection_points[0] - intersection_points[-1])
            widths.append(width)
            # Debug: Draw the line and intersection points
            cv2.drawContours(mask, [contour], -1, 255, thickness=2)
            for pt in intersection_points:
                cv2.circle(mask, tuple(pt), 5, 127, -1)
            cv2.imshow('Debug Mask', mask)
            cv2.waitKey(0)
        else:
            print(f"No intersections found for axis centered at {center}")
    return widths


def calculate_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w / h


def segment_trilete_spores(image, contour, polar_length, widths, width_threshold, aspect_ratio_threshold):
    mask = np.zeros_like(image)
    aspect_ratio = calculate_aspect_ratio(contour)
    if widths and max(widths) > width_threshold and aspect_ratio < aspect_ratio_threshold:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def segment_monolete_spores(image, contour, polar_length, widths, width_threshold, aspect_ratio_threshold):
    mask = np.zeros_like(image)
    aspect_ratio = calculate_aspect_ratio(contour)
    if widths and max(widths) < width_threshold and aspect_ratio >= aspect_ratio_threshold:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def process_spore_image(image, contour):
    polar_axis, polar_length = find_polar_axis(contour)
    equatorial_axes = calculate_equatorial_axes(contour, polar_axis)
    widths = measure_widths(image, contour, equatorial_axes)
    print(f"Measured widths at equatorial axes: {widths}")

    if not widths:
        print("No valid widths found.")
        return polar_length, widths, np.zeros_like(image), np.zeros_like(image)

    width_threshold = max(widths) * 0.7
    # Example value; adjust based on observed characteristics
    aspect_ratio_threshold = 1.5

    trilete_segment = segment_trilete_spores(
        image, contour, polar_length, widths, width_threshold, aspect_ratio_threshold)
    monolete_segment = segment_monolete_spores(
        image, contour, polar_length, widths, width_threshold, aspect_ratio_threshold)

    return polar_length, widths, trilete_segment, monolete_segment


image_list = glob(os.path.join('dataset', '**/*.jpg'), recursive=True)
image_path = random.choice(image_list)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    polar_length, widths, trilete_segment, monolete_segment = process_spore_image(
        image, largest_contour)

    cv2.imshow('Trilete Segmented', trilete_segment)
    cv2.imshow('Monolete Segmented', monolete_segment)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found in the image.")
