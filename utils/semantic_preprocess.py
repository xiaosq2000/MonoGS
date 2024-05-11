from collections import Counter
from PIL import Image
import torch
import torchvision
import numpy as np
import os
from tqdm import tqdm
import pickle


def count_and_label_classes(
    image_path: str, color_to_index={}, min_pixels_per_mask=None
):
    # Read the image
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Flatten the 3D array into a 2D array
    flat_image_array = image_array.reshape(-1, image_array.shape[2])

    # Count occurrences of each color (label)
    color_counts = Counter(map(tuple, flat_image_array))

    # Some masks are too small and inaccurate. Filter them out.
    small_object_color_labels, new_color_labels = [], []
    for color, pixel_count in color_counts.items():
        if min_pixels_per_mask is not None and pixel_count <= min_pixels_per_mask:
            # Filter out colors based on the minimum number of pixels constraint
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


if __name__ == "__main__":
    src_dir = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/dev/projects/MonoGS/datasets/replica_semantic/room0/results_segmentation_maps/Annotations/"
    )
    filenames = sorted(os.listdir(src_dir))
    file_paths = [os.path.join(src_dir, file) for file in filenames]
    dst_dir = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/dev/projects/MonoGS/datasets/replica_semantic/room0/results_segmentation_labels/"
    )
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    segmentation_label_paths = [
        os.path.join(dst_dir, os.path.splitext(file)[0] + ".pt") for file in filenames
    ]

    num_classes = 200
    color_to_index = {}

    for i, file in tqdm(enumerate(file_paths)):
        labeled_image, color_to_index, num_new_label, num_small_objects = (
            count_and_label_classes(file_paths[i], color_to_index)
        )
        tqdm.write(
            f"Index {i}: number of objects = {len(set(color_to_index.values()))}, number of new objects = {num_new_label}, number of small objects = {num_small_objects}"
        )
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

    with open(os.path.join(dst_dir, os.path.pardir, "color_palette.pkl"), "wb") as f:
        pickle.dump(
            {v: k for k, v in color_to_index.items()}, f
        )  # Interchange key and value for convenience

    with open(os.path.join(dst_dir, os.path.pardir, "color_palette.pkl"), "rb") as f:
        dic = pickle.load(f)

    print(dic)
