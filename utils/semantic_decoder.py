import pickle
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


def generate_segmentation_map(decoded_semantics, color_palette, h, w):
    # Get the class indices with highest scores for each pixel
    class_indices = torch.argmax(decoded_semantics, dim=1).cpu().numpy()

    # Create an array of colors corresponding to class indices
    colors = np.array(
        [color_palette.get(label, (0, 0, 0)) for label in range(len(color_palette))],
        dtype=np.uint8,
    )

    # Reshape class indices to match image dimensions
    reshaped_indices = class_indices.reshape(h, w)

    # Index the colors array with class indices to get the segmentation map
    segmentation_map = colors[reshaped_indices]

    return segmentation_map


class SemanticDecoder(nn.Module):
    def __init__(self, semantic_embedding_dim=3, num_classes=200):
        super(SemanticDecoder, self).__init__()
        self.num_classes = num_classes
        self.semantic_embedding_dim = semantic_embedding_dim

        self.fc1 = nn.Linear(self.semantic_embedding_dim, self.num_classes, bias=False)

        # TODO: implement in `dataset.py'
        with open(
            "/mnt/dev-ssd-8T/shuqixiao/data/tum_semantic/rgbd_dataset_freiburg3_long_office_household/color_palette.pkl",
            "rb",
        ) as f:
            self.color_palette = pickle.load(f)
            # self.color_palette = generate_label_colors(self.num_classes)
        self.idx = 0

    def forward(self, x: torch.Tensor):
        # From (c, h, w) to (1, h, w, c)
        x = x.permute(1, 2, 0).unsqueeze(dim=0)
        x = x.view(-1, self.semantic_embedding_dim)
        x = self.fc1(x)
        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        # self.idx = self.idx + 1
        # if self.idx % 10 == 0:
        #     print(self.fc1.weight)
        return x


if __name__ == "__main__":
    target = os.path.normpath(
        "/mnt/dev-ssd-8T/shuqixiao/data/tum_semantic/rgbd_dataset_freiburg3_long_office_household/segmentation_label/1341847980.722988.pt"
    )
    if not os.path.exists(target):
        raise ValueError(f"Error: {target} does not exist.")

    semantic_embedding_dim = 3
    num_classes = 5
    input = torch.randn(
        (semantic_embedding_dim, 960, 1080), dtype=torch.float32, device="cuda"
    )
    semantic_decoder = SemanticDecoder(
        semantic_embedding_dim=semantic_embedding_dim, num_classes=num_classes
    ).to("cuda")
    output = semantic_decoder(input)
    segmentation_map = generate_segmentation_map(
        output, semantic_decoder.color_palette, 960, 1080
    )
    # Display the segmentation map using Matplotlib
    plt.imshow(segmentation_map)
    plt.axis("off")  # Turn off axis
    plt.show()
