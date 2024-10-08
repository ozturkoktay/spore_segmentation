from glob import glob
import os
import random
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred


def segment_spores(image):
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return opened


def compute_lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp


def compute_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[
                        0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, homogeneity, energy


def draw_feature_visualizations(image, trilete_spores, monolete_spores):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for spore, bbox, texture_features in trilete_spores:
        x, y, w, h = bbox
        cv2.drawContours(output_image, [spore], -1, (0, 255, 0), 2)
        solidity = calculate_solidity(spore)
        circularity = calculate_circularity(spore)
        aspect_ratio, _ = calculate_aspect_ratio(spore)
        cv2.putText(output_image, 'Trilete', (x + w // 2, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(output_image, f'S:{solidity:.2f} C:{circularity:.2f} AR:{aspect_ratio:.2f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for spore, bbox, texture_features in monolete_spores:
        x, y, w, h = bbox
        cv2.drawContours(output_image, [spore], -1, (0, 0, 255), 2)
        solidity = calculate_solidity(spore)
        circularity = calculate_circularity(spore)
        aspect_ratio, _ = calculate_aspect_ratio(spore)
        cv2.putText(output_image, 'Monolete', (x + w // 2, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(output_image, f'S:{solidity:.2f} C:{circularity:.2f} AR:{aspect_ratio:.2f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output_image


def calculate_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0


def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0


def calculate_aspect_ratio(contour):
    rect = cv2.minAreaRect(contour)
    width, length = rect[1]  # Get width and length of the bounding box
    aspect_ratio = length / width if width > 0 else 0
    return aspect_ratio, rect


def approximate_contour(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def identify_spores(segmented_image,
                    min_solidity_for_trilete=0.9,
                    min_contour_area=100,
                    min_vertices_for_monolete=7,
                    min_solidity_for_monolete=0.7,
                    max_circularity_for_monolete=0.80,
                    max_aspect_ratio_for_monolete=0.6):
    contours, _ = cv2.findContours(
        segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trilete_spores = []
    monolete_spores = []

    for cnt in contours:
        solidity = calculate_solidity(cnt)
        circularity = calculate_circularity(cnt)
        approx_contour = approximate_contour(cnt)
        num_vertices = len(approx_contour)
        area = cv2.contourArea(cnt)
        aspect_ratio, rect = calculate_aspect_ratio(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        cropped_spore = segmented_image[y:y+h, x:x+w]
        lbp_texture = compute_lbp(cropped_spore)
        glcm_contrast, glcm_homogeneity, glcm_energy = compute_glcm_features(
            cropped_spore)

        print(f'Solidity: {solidity}, Circularity: {circularity}, Vertices: {num_vertices}, '
              f'Aspect Ratio: {aspect_ratio}, GLCM Contrast: {glcm_contrast}, '
              f'Homogeneity: {glcm_homogeneity}, Energy: {glcm_energy}')

        if area < min_contour_area or solidity == 0 or circularity == 0:
            continue

        if w < 25 or h < 25:
            continue

        # Trilete spores: focus primarily on high solidity, relaxed circularity and aspect ratio checks
        if solidity >= min_solidity_for_trilete and num_vertices <= 7:
            trilete_spores.append((cnt, cv2.boundingRect(
                cnt), (lbp_texture, glcm_contrast, glcm_homogeneity, glcm_energy)))

        # Monolete spores: less compact, higher aspect ratio, less circular
        elif (solidity >= min_solidity_for_monolete and
              num_vertices >= min_vertices_for_monolete and
              circularity <= max_circularity_for_monolete and
              aspect_ratio >= max_aspect_ratio_for_monolete):
            monolete_spores.append((cnt, cv2.boundingRect(
                cnt), (lbp_texture, glcm_contrast, glcm_homogeneity, glcm_energy)))

        else:
            print(f"Ambiguous spore with Solidity: {solidity}, Circularity: {circularity}, "
                  f'Aspect Ratio: {aspect_ratio}')

    return trilete_spores, monolete_spores


def visualize_feature(image, spores, bboxes, feature_name, feature_values, color):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for spore, bbox, value in zip(spores, bboxes, feature_values):
        x, y, w, h = bbox
        cv2.drawContours(output_image, [spore], -1, color, 2)
        cv2.putText(output_image, f'{feature_name}: {value:.2f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if feature_name == 'Solidity':
            # Draw the convex hull for solidity visualization
            hull = cv2.convexHull(spore)
            cv2.drawContours(output_image, [hull], -1, (255, 0, 0), 1)
        elif feature_name == 'Circularity':
            M = cv2.moments(spore)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                radius = int(np.sqrt(cv2.contourArea(spore) / np.pi))
                cv2.circle(output_image, (cX + x, cY + y),
                           radius, (255, 0, 0), 2)

        elif feature_name == 'Aspect Ratio':
            rect = cv2.minAreaRect(spore)
            box = cv2.boxPoints(rect)
            box = np.array(box).astype(int)

            # Draw the bounding box
            cv2.drawContours(output_image, [box], 0, (0, 255, 255), 1)

            # Extract details from the rectangle
            (x_center, y_center), (width, height), angle = rect

            # Handle the case when the angle is negative
            if angle < -45:
                angle += 90

            # Compute the angle in radians
            angle_rad = np.deg2rad(angle)

            # Compute the offset for width and height lines based on rotation
            offset_x_width = (width / 2) * np.cos(angle_rad)
            offset_y_width = (width / 2) * np.sin(angle_rad)
            offset_x_height = (height / 2) * np.sin(angle_rad)
            offset_y_height = (height / 2) * np.cos(angle_rad)

            # Compute start and end points for width and height lines
            width_line_start = (int(x_center - offset_x_width),
                                int(y_center - offset_y_width))
            width_line_end = (int(x_center + offset_x_width),
                              int(y_center + offset_y_width))
            height_line_start = (
                int(x_center - offset_x_height), int(y_center + offset_y_height))
            height_line_end = (int(x_center + offset_x_height),
                               int(y_center - offset_y_height))

            # Draw lines for width and height
            cv2.line(output_image, width_line_start,
                     width_line_end, (0, 255, 0), 2)
            cv2.line(output_image, height_line_start,
                     height_line_end, (255, 0, 0), 2)

            # Label the width and height lines
            # Ensure labels are within image boundaries
            label_width_position = (width_line_end[0] + 10, width_line_end[1])
            label_height_position = (
                height_line_end[0] + 10, height_line_end[1] + 20)
            label_width_position = (min(max(label_width_position[0], 0), image.shape[1] - 100),
                                    min(max(label_width_position[1], 0), image.shape[0] - 10))
            label_height_position = (min(max(label_height_position[0], 0), image.shape[1] - 100),
                                     min(max(label_height_position[1], 0), image.shape[0] - 10))

            # Define font parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_color_width = (0, 255, 0)  # Green for width
            text_color_height = (255, 0, 0)  # Red for height

            # Calculate the text size for width and height
            width_text = f'Width: {width:.2f}'
            height_text = f'Height: {height:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                width_text, font, font_scale, font_thickness)
            (text_width_h, text_height_h), _ = cv2.getTextSize(
                height_text, font, font_scale, font_thickness)

            # Draw text background rectangles for better visibility
            cv2.rectangle(output_image, (width_line_start[0], width_line_start[1] - 25), (
                width_line_start[0] + text_width, width_line_start[1]), (0, 0, 0), thickness=cv2.FILLED)
            cv2.rectangle(output_image, (height_line_start[0], height_line_start[1] - 25), (
                height_line_start[0] + text_width_h, height_line_start[1]), (0, 0, 0), thickness=cv2.FILLED)

            # Draw the text on the image
            cv2.putText(output_image, width_text, (width_line_start[0], width_line_start[1] - 5),
                        font, font_scale, text_color_width, font_thickness, lineType=cv2.LINE_AA)
            cv2.putText(output_image, height_text, (height_line_start[0], height_line_start[1] - 5),
                        font, font_scale, text_color_height, font_thickness, lineType=cv2.LINE_AA)

        elif feature_name == 'GLCM Contrast':
            # GLCM features are texture-based, so just annotate with text
            cv2.putText(output_image, f'GLCM Contrast: {value:.2f}',
                        (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return output_image


def draw_individual_feature_visualizations(image, trilete_spores, monolete_spores):
    trilete_solidity = [calculate_solidity(
        spore[0]) for spore in trilete_spores]
    monolete_solidity = [calculate_solidity(
        spore[0]) for spore in monolete_spores]

    trilete_circularity = [calculate_circularity(
        spore[0]) for spore in trilete_spores]
    monolete_circularity = [calculate_circularity(
        spore[0]) for spore in monolete_spores]

    trilete_aspect_ratio = [calculate_aspect_ratio(
        spore[0])[0] for spore in trilete_spores]
    monolete_aspect_ratio = [calculate_aspect_ratio(
        spore[0])[0] for spore in monolete_spores]

    trilete_glcm_contrast = [spore[2][1] for spore in trilete_spores]
    monolete_glcm_contrast = [spore[2][1] for spore in monolete_spores]

    trilete_bboxes = [spore[1] for spore in trilete_spores]
    monolete_bboxes = [spore[1] for spore in monolete_spores]

    feature_sets = [
        ('Solidity', (0, 255, 0), trilete_solidity, monolete_solidity),
        ('Circularity', (0, 0, 255), trilete_circularity, monolete_circularity),
        ('Aspect Ratio', (255, 0, 0), trilete_aspect_ratio, monolete_aspect_ratio),
        ('GLCM Contrast', (255, 255, 0),
         trilete_glcm_contrast, monolete_glcm_contrast)
    ]

    for feature_name, color, trilete_values, monolete_values in feature_sets:
        trilete_image = visualize_feature(image, [spore[0] for spore in trilete_spores],
                                          trilete_bboxes, feature_name, trilete_values, color)
        monolete_image = visualize_feature(image, [spore[0] for spore in monolete_spores],
                                           monolete_bboxes, feature_name, monolete_values, color)

        cv2.imwrite(f'trilete_{feature_name.lower()}.jpg', trilete_image)
        cv2.imwrite(f'monolete_{feature_name.lower()}.jpg', monolete_image)
        print(
            f'Saved {feature_name} visualization for Trilete and Monolete spores.')


def save_feature_visualizations(image, trilete_spores, monolete_spores, filename):
    output_image = draw_feature_visualizations(
        image, trilete_spores, monolete_spores)
    cv2.imwrite(filename, output_image)
    print(f"Saved feature visualization to {filename}")


image_list = glob(os.path.join(
    'dataset/Solo_Spores/', '*.jpg'), recursive=True)
image_path = random.choice(image_list)
print(f'Segmenting the image: {image_path}')
image = preprocess_image(image_path)
segmented = segment_spores(image)
trilete_spores, monolete_spores = identify_spores(segmented)

draw_individual_feature_visualizations(image, trilete_spores, monolete_spores)
output_image = draw_feature_visualizations(
    image, trilete_spores, monolete_spores)
save_feature_visualizations(
    image, trilete_spores, monolete_spores, 'spores_texture_feature_visualization.jpg')

cv2.imshow('Labeled Spores', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
