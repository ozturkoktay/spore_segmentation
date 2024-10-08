from flask import Flask, render_template, request, redirect, url_for
import os
import shutil
import torch
import random
import numpy as np
from torchvision import transforms, models
from PIL import Image
from glob import glob

app = Flask(__name__)
IMAGES_PER_PAGE = 10

# Function to set random seeds for reproducibility


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a random seed
set_random_seed(42)

# Model definition


class SporeModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(SporeModel, self).__init__()
        self.base_model = models.densenet169(pretrained=True)
        self.base_model.classifier = torch.nn.Linear(
            self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Load model
model_path = 'model.pth'
model = SporeModel(num_classes=2)
model.load_state_dict(torch.load(
    model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

BASE_IMAGE_FOLDER = 'static/images/'


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomRotation(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])


def load_images(page, limit=10):
    image_files = glob(os.path.join(
        BASE_IMAGE_FOLDER, '**/*.jpg'), recursive=True)
    start = (page - 1) * limit
    end = start + limit
    image_files = image_files[start:end]
    print(f'Number of images: {len(image_files)}')

    predictions = []
    transform = get_transforms()

    monolete_files = {os.path.basename(f) for f in glob(
        os.path.join('static/images/monolete', '*.jpg'))}
    trilete_files = {os.path.basename(f) for f in glob(
        os.path.join('static/images/trilete', '*.jpg'))}

    for file_path in image_files:
        image_name = os.path.basename(file_path)

        if image_name in monolete_files:
            pred_value, color = '0', 'green'
        elif image_name in trilete_files:
            pred_value, color = '1', 'red'
        else:
            image = Image.open(file_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                confidence, pred = torch.max(output, 1)
                confidence = confidence.item()

            if confidence > 0.5:
                pred_value = str(pred.item())
                color = 'green' if pred_value == '0' else 'red'
                target_folder = os.path.join(
                    'static/images', 'monolete' if pred_value == '0' else 'trilete')

                shutil.move(file_path, os.path.join(
                    target_folder, image_name))

            pred_value, color = 'unknown', 'gray'

        predictions.append(
            {'file_name': image_name, 'pred': pred_value, 'color': color})

    return predictions


def count_images():
    image_files = glob(os.path.join(
        BASE_IMAGE_FOLDER, '**/*.jpg'), recursive=True)
    return len(image_files)


def update_label(image_id, prv_class, new_class):
    base_folder = 'static/images'
    unknown_folder = os.path.join(base_folder, prv_class)
    target_folder = os.path.join(base_folder, new_class)
    shutil.move(os.path.join(unknown_folder, image_id),
                os.path.join(target_folder, image_id))
