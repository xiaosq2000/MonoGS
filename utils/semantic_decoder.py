import torch
import colorsys
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os


def generate_label_colors(num_labels):
    """
    Generate a dictionary mapping index labels to visually distinguishable RGB colors.
    Ensure that label 0 corresponds to black (0, 0, 0).

    Args:
    - num_labels (int): Number of labels (excluding label 0).

    Returns:
    - dict: Dictionary mapping index labels to RGB colors.
    """
    # Assign black color to label 0
    label_colors = {0: (0, 0, 0)}

    # Generate distinct colors for the remaining labels
    if num_labels > 0:
        num_unique_colors = min(
            num_labels, 360
        )  # Limit the number of unique colors to 360 (360 degrees in HSV)
        unique_hues = np.linspace(
            0, 1, num_unique_colors, endpoint=False
        )  # Equally spaced hues
        np.random.shuffle(unique_hues)  # Shuffle the hues for randomness

        # Convert unique hues to RGB colors
        for i, hue in enumerate(unique_hues, start=1):
            rgb = colorsys.hsv_to_rgb(
                hue, 0.9, 0.9
            )  # Using constant saturation and value for clarity
            label_colors[i] = tuple(
                int(c * 255) for c in rgb
            )  # Scale RGB values to [0, 255]

    return label_colors


def generate_segmentation_map(predict, color_palette, h, w):
    masks = torch.argmax(predict, dim=1)
    masks = masks.reshape(1, h, w)
    labeled_image_array = (
        masks.permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.uint8)
    )
    labeled_image_array = np.squeeze(labeled_image_array, axis=2)
    height, width = labeled_image_array.shape

    # Convert the labeled image array to RGB array using label_colors dictionary
    segmentation_map = np.array(
        [color_palette.get(label, (0, 0, 0)) for label in labeled_image_array.flatten()]
    ).astype(np.uint8)
    segmentation_map = segmentation_map.reshape((height, width, 3))

    return segmentation_map


class SemanticDecoder(nn.Module):
    def __init__(self, input_size=3, num_classes=200):
        super(SemanticDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes, bias=True)
        self.num_classes = num_classes
        self.color_palette = generate_label_colors(self.num_classes)
        self.idx = 0

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 2, 0).unsqueeze(dim=0)
        x = x.view(-1, 3)
        x = self.fc1(x)
        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        # self.idx = self.idx + 1
        # if self.idx % 10 == 0:
        #     print(self.fc1.weight)
        return x

    def generate_masks(self, x):
        _, h, w = x.shape
        x = torch.argmax(self.forward(x), dim=1)
        x = x.reshape(1, h, w)
        return x

    def generate_segmentation_map(self, x):
        masks = self.generate_masks(x)
        labeled_image_array = (
            masks.permute(1, 2, 0).squeeze(0).cpu().numpy().astype(np.uint8)
        )
        labeled_image_array = np.squeeze(labeled_image_array, axis=2)
        height, width = labeled_image_array.shape

        # Convert the labeled image array to RGB array using label_colors dictionary
        segmentation_map = np.array(
            [
                self.color_palette.get(label, (0, 0, 0))
                for label in labeled_image_array.flatten()
            ]
        ).astype(np.uint8)
        segmentation_map = segmentation_map.reshape((height, width, 3))

        return segmentation_map


if __name__ == "__main__":
    target = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/data/tum_semantic/rgbd_dataset_freiburg3_long_office_household/segmentation_label/1341847980.722988.pt"
    )
    if not os.path.exists(target):
        raise ValueError(f"Error: {target} does not exist.")

    num_classes = 5
    input = torch.randn((3, 960, 1080), dtype=torch.float32, device="cuda")
    semantic_decoder = SemanticDecoder(num_classes=num_classes).to("cuda")
    output = semantic_decoder(input)
    masks = semantic_decoder.generate_masks(input)
    segmentation_map = semantic_decoder.generate_segmentation_map(input)
    # Display the segmentation map using Matplotlib
    plt.imshow(segmentation_map)
    plt.axis("off")  # Turn off axis
    plt.show()
