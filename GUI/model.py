import pandas as pd
from scipy.stats import binom
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
import torch
import random
import numpy as np
from torchvision import transforms, models
from PIL import Image
from glob import glob
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torchvision.models import ResNet152_Weights, ResNet18_Weights
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (for non-GUI rendering)

LOWER_THRESHOLD = 0.30
UPPER_THRESHOLD = 0.70
BASE_IMAGE_FOLDER = 'static/images/'
TEMPERATURE = 0.5  # Adjust based on calibration

# Global variable to store all predictions
all_predictions = []


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a random seed
set_random_seed(32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model definition


class SporeModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(SporeModel, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.Dropout(0.5)

        )

    def forward(self, x):
        return self.model(x)


class TemperatureScaling:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, logits):
        return logits / self.temperature


scaler = TemperatureScaling(TEMPERATURE)

# model_path = 'model.pth'
model_path = 'resnet18_model.pth'
model = SporeModel(num_classes=2)
model.load_state_dict(torch.load(
    model_path, map_location=device, weights_only=True), strict=False)
model.eval()


def save_scripted_model():
    scripted_model = torch.jit.script(model)  # Convert to TorchScript
    # Save the scripted model
    torch.jit.save(scripted_model, 'scripted_model.pth')

# Function to load the TorchScript model


def load_scripted_model():
    if os.path.exists('scripted_model.pth'):
        print("Loading the pre-saved scripted model...")
        scripted_model = torch.jit.load(
            'scripted_model.pth', map_location=device)
        scripted_model.eval()  # Put the model in evaluation mode
        return scripted_model
    else:
        print("Scripted model not found, saving it now...")
        save_scripted_model()  # Save the model if it doesn't exist
        return load_scripted_model()  # Load the saved model after saving


# Load or save the TorchScript model
model = load_scripted_model()


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


confidence_scores = []
confidence_monolete = []
confidence_trilete = []


def make_predictions():
    image_files = glob(os.path.join(
        BASE_IMAGE_FOLDER, '**/*.jpg'), recursive=True)
    print(f"Total images found: {len(image_files)}")  # Debugging print

    random.shuffle(image_files)
    predictions = []
    transform = get_transforms()

    class_folders = {
        'monolete': set(),
        'trilete': set(),
        'unknown': set(),
    }

    for folder in glob(os.path.join(BASE_IMAGE_FOLDER, '*')):
        if os.path.isdir(folder):
            class_name = os.path.basename(folder)
            class_folders[class_name].update(
                os.path.basename(f) for f in glob(os.path.join(folder, '*.jpg')))

    def process_image(file_path):
        image_name = os.path.basename(file_path)
        class_name = 'unknown'
        color = 'gray'
        wrong = False

        if image_name in class_folders['monolete']:
            class_name, color = 'monolete', 'green'
            confidence = 1.0  # Assume it's high-confidence if it's pre-labeled
        elif image_name in class_folders['trilete']:
            class_name, color = 'trilete', 'red'
            confidence = 1.0  # Assume it's high-confidence if it's pre-labeled
        else:
            # Model inference for new image
            image = Image.open(file_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                output = scaler(output)  # Apply temperature scaling
                # Use sigmoid for binary classification
                probabilities = F.sigmoid(output)
                # Get the class with highest confidence
                confidence, pred = torch.max(probabilities, 1)
                confidence = confidence.item()

            confidence_scores.append(confidence)

            # if pred.item() == 0:
            #     class_name, color = 'monolete', 'green'
            #     confidence_monolete.append(confidence)
            # elif pred.item() == 1:
            #     class_name, color = 'trilete', 'red'
            #     confidence_trilete.append(confidence)

            if confidence <= LOWER_THRESHOLD:
                # Mark this prediction as uncertain or wrong
                wrong = True
                class_name = 'unknown'
            elif confidence >= UPPER_THRESHOLD and pred.item() == 0:
                # Confidently classified as 'monolete'
                class_name, color = 'monolete', 'green'
                target_folder = os.path.join('static/images', 'monolete')
                shutil.move(file_path, os.path.join(target_folder, image_name))
                confidence_monolete.append(confidence)
            elif confidence >= UPPER_THRESHOLD and pred.item() == 1:
                # Confidently classified as 'trilete'
                class_name, color = 'trilete', 'red'
                target_folder = os.path.join('static/images', 'trilete')
                shutil.move(file_path, os.path.join(target_folder, image_name))
                confidence_trilete.append(confidence)

        # ================================================================

        # if image_name in class_folders['monolete']:
        #     class_name, color = 'monolete', 'green'
        #     confidence = 0.0
        # elif image_name in class_folders['trilete']:
        #     class_name, color = 'trilete', 'red'
        #     confidence = 1.0
        # else:
        #     image = Image.open(file_path).convert('RGB')
        #     image = transform(image).unsqueeze(0)

        #     with torch.no_grad():
        #         output = model(image)
        #         output = scaler(output)
        #         # probabilities = F.softmax(scaler_output, dim=1)
        #         probabilities = F.sigmoid(output)
        #         confidence, pred = torch.max(probabilities, 1)
        #         confidence = confidence.item()
        #     confidence_scores.append(confidence)

        #     if confidence <= LOWER_THRESHOLD:
        #         # Image is confidently classified as 'monolete'
        #         class_name, color = 'monolete', 'green'
        #         target_folder = os.path.join('static/images', 'monolete')
        #         shutil.move(file_path, os.path.join(target_folder, image_name))
        #         confidence_monolete.append(confidence)
        #     elif confidence >= UPPER_THRESHOLD:
        #         # Image is confidently classified as 'trilete'
        #         class_name, color = 'trilete', 'red'
        #         target_folder = os.path.join('static/images', 'trilete')
        #         shutil.move(file_path, os.path.join(target_folder, image_name))
        #         confidence_trilete.append(confidence)

        #     else:
        #         wrong = True
        #         class_name = 'unknown'
            # print(f"Wrong prediction: {file_path}, confidence: {confidence}")

        return {'file_name': image_name, 'pred': class_name, 'color': color, 'confidence': confidence, 'wrong': wrong}

    # def process_image(file_path):
    #     image_name = os.path.basename(file_path)
    #     class_name = 'unknown'
    #     color = 'gray'

    #     if image_name in class_folders['monolete']:
    #         class_name, color = '0', 'green'
    #     elif image_name in class_folders['trilete']:
    #         class_name, color = '1', 'red'
    #     else:
    #         image = Image.open(file_path).convert('RGB')
    #         image = transform(image).unsqueeze(0)

    #         with torch.no_grad():
    #             output = model(image)
    #             probabilities = F.softmax(output, dim=1)
    #             confidence, pred = torch.max(probabilities, 1)
    #             confidence = confidence.item()

    #         if confidence > 0.5:
    #             class_name = str(pred.item())
    #             color = 'green' if class_name == '0' else 'red'
    #             target_folder = os.path.join(
    #                 'static/images', 'monolete' if class_name == '0' else 'trilete')
    #             shutil.move(file_path, os.path.join(target_folder, image_name))

    #     return {'file_name': image_name, 'pred': class_name, 'color': color}
    num_workers = os.cpu_count() // 2
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        predictions = list(executor.map(process_image, image_files))

    monolete_lower = np.percentile(confidence_monolete, 5)  # 5th percentile
    monolete_upper = np.percentile(confidence_monolete, 95)  # 95th percentile

    trilete_lower = np.percentile(confidence_trilete, 5)  # 5th percentile
    trilete_upper = np.percentile(confidence_trilete, 95)  # 95th percentile

    OVERLAP_LOWER_THRESHOLD = max(monolete_lower, trilete_lower)
    OVERLAP_UPPER_THRESHOLD = min(monolete_upper, trilete_upper)

    if OVERLAP_LOWER_THRESHOLD < OVERLAP_UPPER_THRESHOLD:
        print(
            f"Overlap range: {OVERLAP_LOWER_THRESHOLD:.2f} to {OVERLAP_UPPER_THRESHOLD:.2f}")
    else:
        print("No significant overlap")

    plot_confidence_scores(confidence_scores, LOWER_THRESHOLD, UPPER_THRESHOLD)
    plot_confidence_score_distributions(
        confidence_monolete, confidence_trilete)
    plot_binomial(confidence_monolete, confidence_trilete)

    return predictions


# def plot_confidence_distribution(confidence_monolete, confidence_trilete, lower_threshold, upper_threshold):
#     sns.set(style="whitegrid")

#     plt.figure(figsize=(14, 6))

#     # Check if the confidence lists are not empty
#     if confidence_monolete:
#         ci_monolete = np.percentile(confidence_monolete, [2.5, 97.5])
#         sns.histplot(confidence_monolete, bins=50, alpha=0.7,
#                      color='green', label='Monolete')
#         plt.axvline(x=ci_monolete[0], color='green', linestyle='--')
#         plt.axvline(x=ci_monolete[1], color='green', linestyle='--')
#     else:
#         print("No Monolete predictions")

#     if confidence_trilete:
#         ci_trilete = np.percentile(confidence_trilete, [2.5, 97.5])
#         sns.histplot(confidence_trilete, bins=50, alpha=0.7,
#                      color='red', label='Trilete')
#         plt.axvline(x=ci_trilete[0], color='red', linestyle='--')
#         plt.axvline(x=ci_trilete[1], color='red', linestyle='--')
#     else:
#         print("No Trilete predictions")

#     # Add threshold lines with annotations
#     plt.axvline(x=lower_threshold, color='blue', linestyle='--',
#                 linewidth=2, label='Lower Threshold')
#     plt.axvline(x=upper_threshold, color='orange', linestyle='--',
#                 linewidth=2, label='Upper Threshold')

#     # Titles and labels
#     plt.title('Confidence Score Distribution', fontsize=16)
#     plt.xlabel('Confidence Score', fontsize=14)
#     plt.ylabel('Frequency', fontsize=14)

#     # Show legend and grid
#     plt.legend(loc='upper left')
#     plt.grid(True)

#     # Save plot to a file
#     plt.tight_layout()
#     plt.savefig('confidence_distribution.png')  # Save as PNG
#     plt.close()


def plot_confidence_score_distributions(confidence_monolete, confidence_trilete):
    plt.figure(figsize=(10, 6))

    # Plot histograms for both confidence scores
    hist_monolete, bins_monolete = np.histogram(
        confidence_monolete, bins=20, density=True)
    hist_trilete, bins_trilete = np.histogram(
        confidence_trilete, bins=20, density=True)

    bin_centers_monolete = 0.5 * (bins_monolete[:-1] + bins_monolete[1:])
    bin_centers_trilete = 0.5 * (bins_trilete[:-1] + bins_trilete[1:])

    plt.bar(bin_centers_monolete, hist_monolete, width=(bins_monolete[1] - bins_monolete[0]), alpha=0.7, color='green',
            label='Monolete Confidence Scores', edgecolor='black')
    plt.bar(bin_centers_trilete, hist_trilete, width=(bins_trilete[1] - bins_trilete[0]), alpha=0.7, color='red',
            label='Trilete Confidence Scores', edgecolor='black')

    # Determine overlap
    min_confidence = max(min(confidence_monolete), min(confidence_trilete))
    max_confidence = min(max(confidence_monolete), max(confidence_trilete))

    if min_confidence < max_confidence:
        # Recalculate histograms with same bins for overlap region
        overlap_bins = np.linspace(min_confidence, max_confidence, 20)
        overlap_hist_monolete, _ = np.histogram(
            confidence_monolete, bins=overlap_bins, density=True)
        overlap_hist_trilete, _ = np.histogram(
            confidence_trilete, bins=overlap_bins, density=True)

        overlap_x = 0.5 * (overlap_bins[:-1] + overlap_bins[1:])
        overlap_y = np.minimum(overlap_hist_monolete, overlap_hist_trilete)

        plt.fill_between(overlap_x, overlap_y, color='purple',
                         alpha=0.5, label='Overlap Area')

    else:
        plt.text(0.5, 0.5, 'No Overlap Detected', transform=plt.gca().transAxes,
                 fontsize=14, color='purple', horizontalalignment='center')

    plt.title('Confidence Score Distribution for Monolete and Trilete', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('confidence_score_distribution_monolete_trilete.png')
    plt.close()


def plot_binomial(confidence_monolete, confidence_trilete):
    # Convert lists to numpy arrays for element-wise operations
    monolete_lower = np.percentile(confidence_monolete, 5)  # 5th percentile
    monolete_upper = np.percentile(confidence_monolete, 95)  # 95th percentile

    trilete_lower = np.percentile(confidence_trilete, 5)  # 5th percentile
    trilete_upper = np.percentile(confidence_trilete, 95)  # 95th percentile

    OVERLAP_LOWER_THRESHOLD = max(monolete_lower, trilete_lower)
    OVERLAP_UPPER_THRESHOLD = min(monolete_upper, trilete_upper)

    # Print calculated thresholds for debug
    print(f"Monolete lower: {monolete_lower}, upper: {monolete_upper}")
    print(f"Trilete lower: {trilete_lower}, upper: {trilete_upper}")
    print(f"Overlap lower threshold: {OVERLAP_LOWER_THRESHOLD}")
    print(f"Overlap upper threshold: {OVERLAP_UPPER_THRESHOLD}")

    n_monolete = len(confidence_monolete)
    n_trilete = len(confidence_trilete)

    # Calculate success probability for both groups based on the overlap thresholds
    successes_monolete = sum([1 for score in confidence_monolete if score >=
                             OVERLAP_LOWER_THRESHOLD and score <= OVERLAP_UPPER_THRESHOLD])
    p_success_monolete = successes_monolete / n_monolete if n_monolete > 0 else 0

    successes_trilete = sum([1 for score in confidence_trilete if score >=
                            OVERLAP_LOWER_THRESHOLD and score <= OVERLAP_UPPER_THRESHOLD])
    p_success_trilete = successes_trilete / n_trilete if n_trilete > 0 else 0

    x_monolete = np.arange(0, n_monolete + 1)
    x_trilete = np.arange(0, n_trilete + 1)

    binom_pmf_monolete = binom.pmf(x_monolete, n_monolete, p_success_monolete)
    binom_pmf_trilete = binom.pmf(x_trilete, n_trilete, p_success_trilete)

    # Plot binomial distributions
    plt.figure(figsize=(10, 6))
    plt.bar(x_monolete, binom_pmf_monolete, alpha=0.7, color='green',
            label=f'Monolete (p={p_success_monolete:.2f})', edgecolor='black')
    plt.bar(x_trilete, binom_pmf_trilete, alpha=0.7, color='red',
            label=f'Trilete (p={p_success_trilete:.2f})', edgecolor='black')

    plt.title(
        'Binomial Distribution of Monolete and Trilete Confidence Scores', fontsize=16)
    plt.xlabel('Number of Successful Classifications', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig('overlap_dist.png')
    plt.close()


def plot_confidence_scores(data, lower_threshold, upper_threshold):
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=50, alpha=0.8, color='skyblue',
             edgecolor='black', label='Confidence Scores')

    # Add threshold lines with annotations
    plt.axvline(x=LOWER_THRESHOLD, color='red', linestyle='--',
                linewidth=2, label='Lower Threshold')
    plt.axvline(x=UPPER_THRESHOLD, color='green', linestyle='--',
                linewidth=2, label='Upper Threshold')

    plt.text(LOWER_THRESHOLD + 0.01, plt.ylim()
             [1] * 0.9, 'Lower Threshold', color='red', fontsize=12, rotation=90)
    plt.text(UPPER_THRESHOLD + 0.01, plt.ylim()
             [1] * 0.9, 'Upper Threshold', color='green', fontsize=12, rotation=90)

    plt.title('Confidence Score Distribution', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('overall_conf_dist.png')
    plt.close()


def initialize_predictions():
    global all_predictions
    all_predictions = make_predictions()
    # Debugging print
    print(f"All predictions initialized: {len(all_predictions)}")
    return all_predictions


# def get_paginated_predictions(page, limit=10, class_filter=None):
#     global all_predictions
#     if class_filter:
#         filtered_predictions = [
#             p for p in all_predictions if p['pred'] == class_filter]
#     else:
#         filtered_predictions = all_predictions
#     start = (page - 1) * limit
#     end = start + limit
#     return filtered_predictions[start:end]

def get_paginated_predictions(page, limit=10, class_filter=None):
    # Filter only by class, no confidence filtering

    if class_filter:
        filtered_predictions = [
            p for p in all_predictions if p['pred'] == class_filter]
    else:
        filtered_predictions = all_predictions
    # Pagination
    start = (page - 1) * limit
    end = start + limit
    return filtered_predictions[start:end], len(filtered_predictions)


def count_images(class_filter=None):
    global all_predictions
    if class_filter:
        return len([p for p in all_predictions if p['pred'] == class_filter])
    return len(all_predictions)


# def get_paginated_predictions(page, limit=10):
#     start = (page - 1) * limit
#     end = start + limit
#     return all_predictions[start:end]


# def count_images():
#     return len(all_predictions)


def update_label(image_id, prv_class, new_class):
    base_folder = 'static/images'
    # Move the image to the new class folder
    src_folder = os.path.join('static/images', prv_class)
    dest_folder = os.path.join('static/images', new_class)

    # Move the image to the correct folder
    src_file = os.path.join(src_folder, image_id)
    dest_file = os.path.join(dest_folder, image_id)

    if os.path.exists(src_file):
        shutil.move(src_file, dest_file)

    # Update prediction in the all_predictions list
    for prediction in all_predictions:
        if prediction['file_name'] == image_id:
            prediction['pred'] = new_class
            # Set wrong to False since the user corrected it
            prediction['wrong'] = False
            break
