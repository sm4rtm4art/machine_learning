import math
import random

import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import rotate

from ml_portfolio.common.paths import get_project_paths

"""
Generate randomized 2D samples with exactly on distinct feature
"""
PROJECT_NAME = "vision_ssl_transfer"


def generate(n_features, sample_length, sample_number, noise_strength=0.05):
    x, y = generate_positions(sample_length)

    samples = np.zeros([sample_number, sample_length, sample_length])

    for i in range(sample_number):
        samples[i] = generate_random_feature(x, y, n_features, noise_strength)
    return samples


def generate_positions(sample_length):
    x = np.linspace(0, 1, sample_length)
    return np.meshgrid(x, x)


def gaussian(x, y, peak_position, peak_width):
    return np.exp(
        -((x - peak_position[0]) ** 2 + (y - peak_position[1]) ** 2) / (2 * peak_width ** 2)
    )


def sin(x, y, frequency, phase):
    return np.sin(2 * math.pi * frequency[0] * x + phase[0]) + np.sin(
        2 * math.pi * frequency[1] * y + phase[1]
    )


#
# """
# features are broad gaussian peak, sharp gaussian peak, oscillations, and no feature (pure noise)
# """
#
def generate_random_feature(x, y, n_features, noise_strength=0.01):
    sample = np.zeros_like(x)
    feature_id = random.randint(0, n_features - 1)
    match feature_id:
        # Random Gaussian peak
        case 0:
            peak_position = 0.25 + 0.5 * np.random.random(2)
            peak_width = 0.15
            sample += gaussian(x, y, peak_position, peak_width)

        # Random sharp Gaussian peak
        case 1:
            peak_position = 0.25 + 0.5 * np.random.random(2)
            peak_width = 0.05
            sample += gaussian(x, y, peak_position, peak_width)

        # sinus
        case 2:
            phase = 2 * math.pi * np.random.random(2)
            frequency = 1 + 10 * np.random.random(2)
            sample += sin(x, y, frequency, phase)

        # triangle
        case 3:
            center = 0.25 + 0.25 * np.random.random(2)
            base = 0.2 + 0.5 * random.random()
            height = 0.2 + 0.5 * random.random()
            width = 0.05 + 0.05 * random.random()
            sample = triangle(x, y, center, base, height)

        #  elipse
        case 4:
            center = 0.25 + 0.25 * np.random.random(2)
            radius = 0.2 + 0.2 * random.random()
            width = 0.01 + 0.01 * random.random()
            xa = 1
            ya = 1.5
            sample = elipse(x, y, center, radius, xa, ya, width)

        case 5:
            center = 0.25 + 0.25 * np.random.random(2)
            length = 0.2 + 0.5 * random.random()
            height = 0.2 + 0.5 * random.random()
            width = 0.1 + 0.2 * random.random()
            sample = rectangle(x, y, center, length, height, width)

        case 6:
            sample_length = x.shape[0]
            target_size = (random.randint(20, 128), random.randint(20, 128))
            position = (0.25 + 0.5 * random.random(), 0.25 + 0.5 * random.random())
            sample = turtle(sample_length, target_size, position)

    #  rotate
    angle = 360 * random.random()
    sample = rotate(sample, angle, reshape=False)

    # Add noise
    sample += noise_strength * np.random.randn(*x.shape)

    return sample.astype(np.float32)


def circle(x, y, center, radius, width=0.05):
    sample = width - np.abs(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - radius)
    return np.heaviside(sample, 0)


def elipse(x, y, center, radius, xa, ya, width=0.05):
    sample = width - np.abs(
        np.sqrt(((x - center[0]) / xa) ** 2 + ((y - center[1]) / ya) ** 2) - radius
    )
    return np.heaviside(sample, 0)


def triangle_filled(x, y, center, base, height):
    cx, cy = center
    # Bound Y: between base (cy) and tip (cy + height)
    within_y = np.heaviside(y - cy, 1) * np.heaviside((cy + height) - y, 1)

    # Bound X: linear slope based on height
    # Half-width decreases linearly as y goes from cy to cy + height
    slope_limit = (base / 2) * (1 - (y - cy) / height)
    within_x = np.heaviside(slope_limit - np.abs(x - cx), 1)

    return within_y * within_x


def triangle(x, y, center, base, height, width=0.05):
    # Outer solid triangle
    outer = triangle_filled(x, y, center, base, height)

    # Inner triangle: We scale height/base and shift the center up
    # slightly to keep the 'hollow' border uniform.
    inner_base = base - (width * 2)
    inner_height = height - (width * 2)  # Adjusting for geometry

    # Shift the inner triangle up slightly so the bottom border matches 'width'
    inner_center = (center[0], center[1] + width)

    inner = triangle_filled(x, y, inner_center, inner_base, inner_height)

    # Subtract to get the outline
    sample = outer - inner
    return np.clip(sample, 0, 1)  # Ensure no negative values


def rectangle_filled(x, y, center, length, height):
    cx, cy = center

    # 1. Horizontal bounds: x must be between (cx - L/2) and (cx + L/2)
    # We use (1 - step) for the upper bound to 'cut off' the signal
    within_x = np.heaviside(x - (cx - length / 2), 1) * np.heaviside((cx + length / 2) - x, 1)

    # 2. Vertical bounds: y must be between (cy - H/2) and (cy + H/2)
    within_y = np.heaviside(y - (cy - height / 2), 1) * np.heaviside((cy + height / 2) - y, 1)

    # 3. Intersection (Logical AND)
    sample = within_x * within_y

    return sample


def rectangle(x, y, center, length, height, width=0.05):
    sample = rectangle_filled(x, y, center, length, height)
    sample -= rectangle_filled(x, y, center, length - width / 2, height - width / 2)
    return sample


def turtle(sample_length, target_size, position=(0, 0), threshold=100):
    img = Image.open("turtle.png").convert("L")

    # 2. Squeeze to given size
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    # 3. Make it binary (0 or 1)
    img_array = np.array(img)
    binary_img = (img_array < threshold).astype(np.float32)

    # 4. Create the NxN canvas
    sample = np.zeros((sample_length, sample_length), dtype=np.float32)

    # 5. Place it (with boundary safety)
    y, x = int(position[0]), int(position[1])
    w, h = int(target_size[0]), int(target_size[1])

    # Define the slice limits (clamped to canvas boundaries)
    y_end = min(y + h, sample_length)
    x_end = min(x + w, sample_length)

    # Slice the binary image to fit the available space if it goes off-edge
    sample[y:y_end, x:x_end] = binary_img[: y_end - y, : x_end - x]

    return sample



def generate_and_save_images(
        img_train_dir_base ="images_train",
        img_test_dir_base ="images_test",
        sample_number=128,
        sample_length=128,
        n_features=8,
):
    output_dir = get_project_paths(PROJECT_NAME).data_dir / "prototypes/"
    output_dir.mkdir(exist_ok=True)

    output_dir = output_dir / "ssl_2d_images"

    output_dir.mkdir(exist_ok=True)

    img_train_dir = output_dir / img_train_dir_base
    img_test_dir = output_dir / img_test_dir_base

    # create folder, update time stamp if exists
    img_train_dir.mkdir(exist_ok=True)
    # delete folder and content
    shutil.rmtree(img_train_dir)
    #  create empty folder
    img_train_dir.mkdir()

    img_test_dir.mkdir(exist_ok=True)
    shutil.rmtree(img_test_dir)
    img_test_dir.mkdir()

    annotations_file = "annotations.txt"

    for image_dir in [img_train_dir, img_test_dir]:
        samples = generate(n_features, sample_length, sample_number)
        annotations_file_path = image_dir / annotations_file
        annotations_string = "file name, label id\n"
        #
        for i in range(sample_number):
            # print(str(i))
            file_name = str(i) + ".jpg"
            file_path = image_dir / file_name
            annotations_string += file_name + ', 0'
            #  add line break except for last line (although empty lines are typically ignored)
            if i < sample_number - 1:
                annotations_string += "\n"
            sample = samples[i]

            sample = sample - np.min(sample)
            sample = sample / np.max(sample)
            sample = (255 * sample).astype(np.uint8)
            #
            img = Image.fromarray(sample)

            img.save(file_path, "JPEG")

        with open(annotations_file_path, "w") as file:
            file.write(annotations_string)

    return img_train_dir, img_test_dir, annotations_file
