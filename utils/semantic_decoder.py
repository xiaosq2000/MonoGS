import pickle
import torch
import colorsys
import matplotlib.pyplot as plt
import numpy as np
from torch import nn


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
    class_indices = torch.argmax(decoded_semantics.detach(), dim=1).cpu().numpy()

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


def generate_confidence_map(softmax_decoded_semantics: torch.Tensor, color_map, h, w):
    try:
        generate_confidence_map.idx += 1
    except AttributeError:
        generate_confidence_map.idx = 0

    # Get the maximum softmax probability for each pixel
    max_probs = np.max(softmax_decoded_semantics.detach().cpu().numpy(), axis=1)

    max_probs_map = max_probs.reshape(h, w)

    # Generate confidence map without alpha channel
    confidence_map_rgba = color_map(max_probs_map)
    confidence_map_rgb = confidence_map_rgba[:, :, :3]  # Remove alpha channel
    confidence_map = (confidence_map_rgb * 255).astype(np.uint8)

    # print(generate_confidence_map.idx)
    # if generate_confidence_map.idx == 20:
    #     np.save("/tmp/max_probs_map.pt", max_probs_map)
    #     print("Save to /tmp/max_probs_map.pt")

    return confidence_map


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        semantic_embedding_dim,
        num_classes,
        color_palette_path=None,
        color_palette=None,
    ):
        super(SemanticDecoder, self).__init__()
        self.num_classes = num_classes
        self.semantic_embedding_dim = semantic_embedding_dim

        self.fc1 = nn.Linear(self.semantic_embedding_dim, self.num_classes, bias=False)

        if color_palette_path is not None:
            with open(
                color_palette_path,
                "rb",
            ) as f:
                self.color_palette = pickle.load(f)
        if color_palette is not None:
            self.color_palette = color_palette
        if color_palette is None and color_palette_path is None:
            self.color_palette = generate_label_colors(num_labels=self.num_classes)

        self.heat_map = plt.cm.get_cmap("hot")

    def forward(self, x: torch.Tensor):
        # From (c, h, w) to (1, h, w, c)
        x = x.permute(1, 2, 0).unsqueeze(dim=0)
        x = x.view(-1, self.semantic_embedding_dim)
        x = self.fc1(x)
        x = x.unsqueeze(dim=0).permute(0, 2, 1)
        return x


if __name__ == "__main__":
    semantic_embedding_dim = 16
    num_classes = 5
    h = 960
    w = 1080
    input = torch.randn(
        (semantic_embedding_dim, h, w), dtype=torch.float32, device="cuda"
    )
    semantic_decoder = SemanticDecoder(
        semantic_embedding_dim=semantic_embedding_dim, num_classes=num_classes
    ).to("cuda")
    output = semantic_decoder(input)
    segmentation_map = generate_segmentation_map(
        output, semantic_decoder.color_palette, h, w
    )
    # Display the segmentation map using Matplotlib
    plt.imshow(segmentation_map)
    plt.axis("off")  # Turn off axis
    plt.show()
