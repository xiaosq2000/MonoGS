from collections import Counter
from PIL import Image
import torch
import torchvision
import numpy as np
import os
import random
import colorsys
from tqdm import tqdm


def count_and_label_classes(image_path: str, color_to_index={}, min_pixels_per_mask=None):
    # Read the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the 3D array into a 2D array
    flat_image_array = image_array.reshape(-1, image_array.shape[2])

    # Count occurrences of each color
    color_counts = Counter(map(tuple, flat_image_array))

    # Filter out colors based on the minimum number of pixels constraint
    small_object_color_labels, new_color_labels = [], []

    for color, pixel_count in color_counts.items():
        if min_pixels_per_mask is not None and pixel_count <= min_pixels_per_mask:
            small_object_color_labels.append(color)
        else:
            if color not in color_to_index or color_to_index[tuple(color)] == 0:
                new_color_labels.append(color)

    # Create a dictionary mapping colors to indices
    if color_to_index is None:
        # Assign a unique label to each color
        index_labels = np.arange(1, len(new_color_labels) + 1)
        color_to_index = {
            tuple(color): label for color, label in zip(new_color_labels, index_labels)
        }
    else:
        previous_count = len(set(color_to_index.values()))
        for i, color in enumerate(new_color_labels):
            color_to_index[tuple(color)] = i + previous_count

    # Assign label 0 for colors corresponding to small objects
    for color in small_object_color_labels:
        color_to_index[tuple(color)] = 0

    # Replace the colors in the original image with their corresponding labels
    labeled_image_array = np.array(
        [[color_to_index.get(tuple(pixel), 0) for pixel in row] for row in image_array]
    )

    # Convert the labeled image array back to PIL image
    labeled_image = Image.fromarray(labeled_image_array.astype(np.uint8))

    # Return the labeled image and the number of classes
    return (
        labeled_image,
        color_to_index,
        len(new_color_labels),
        len(small_object_color_labels),
    )


def encode_one_hot(labeled_image, num_classes=-1):
    # Convert PIL image to PyTorch tensor
    labeled_image_tensor = torchvision.transforms.functional.pil_to_tensor(
        labeled_image
    )

    # Use PyTorch's built-in one-hot encoding functionality
    return torch.nn.functional.one_hot(
        labeled_image_tensor.to(torch.int64), num_classes
    )


def generate_colorful_map(labeled_image_array, label_colors):
    """
    Generate a colorful map based on pixel-wise index labels.

    Args:
    - labeled_image_array (numpy.ndarray): Array representing the labeled image with index labels.
    - label_colors (dict): Dictionary mapping index labels to RGB colors.

    Returns:
    - PIL.Image.Image: A colorful map image.
    """
    # Get the height and width of the labeled image array
    height, width = labeled_image_array.shape

    # Create an empty array to store the RGB values of the colorful map
    colorful_map_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over each pixel in the labeled image array and assign the corresponding color
    for i in range(height):
        for j in range(width):
            index_label = labeled_image_array[i, j]
            color = label_colors.get(
                index_label, (0, 0, 0)
            )  # Default to black for unknown labels
            colorful_map_array[i, j] = color

    # Convert the colorful map array to PIL image
    colorful_map_image = Image.fromarray(colorful_map_array)

    return colorful_map_image


def generate_random_label_colors(num_classes=24, is_random=False):
    """
    Generate random RGB colors for index labels.

    Args:
    - num_classes (int): Number of classes or index labels.

    Returns:
    - dict: Dictionary mapping index labels to RGB colors.
    """
    # Generate a list of distinct hues
    hues = [i / num_classes for i in range(num_classes)]
    if is_random:
        random.shuffle(hues)

    # Convert hues to RGB colors
    colors = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in hues]

    # Scale RGB values to 0-255 range and convert to integers
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in colors]

    # Ensure label 0 is black
    colors.insert(0, (0, 0, 0))

    # Initialize an empty dictionary to store the mapping
    color_palette = {}

    # Generate a mapping from index labels to randomly selected colors
    for i in range(num_classes):
        color_palette[i] = colors[i]

    return color_palette


if __name__ == "__main__":
    src_dir = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/data/tum_semantic/rgbd_dataset_freiburg3_long_office_household/segmentation_map"
    )
    filenames = sorted(os.listdir(src_dir))
    file_paths = [os.path.join(src_dir, file) for file in filenames]
    dst_dir = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/data/tum_semantic/rgbd_dataset_freiburg3_long_office_household/segmentation_label"
    )
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    segmentation_label_paths = [
        os.path.join(dst_dir, os.path.splitext(file)[0] + ".pt") for file in filenames
    ]

    color_to_index = {}
    num_classes = 200
    color_palette = generate_random_label_colors(num_classes)

    for i, file in tqdm(enumerate(file_paths)):
        labeled_image, color_to_index, num_new_label, num_small_objects = (
            count_and_label_classes(file_paths[i], color_to_index)
        )
        tqdm.write(
            f"Index {i}: number of objects = {len(set(color_to_index.values())) - 1}, number of new objects = {num_new_label}, number of small objects = {num_small_objects}"
        )
        # one_hot_image = encode_one_hot(labeled_image, num_classes)
        labeled_image_tensor = torchvision.transforms.functional.pil_to_tensor(
            labeled_image
        )
        labeled_image_tensor = labeled_image_tensor.to(torch.int64)
        torch.save(labeled_image_tensor, segmentation_label_paths[i])

        # print(
        #     # sort by index
        #     {k: v for k, v in sorted(color_to_index.items(), key=lambda item: item[1])}
        # )

        # generate_colorful_map(np.array(labeled_image), color_palette).show()

    # print(one_hot_encoding.shape)
