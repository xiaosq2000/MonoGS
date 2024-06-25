import pickle
import torch
import colorsys
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from utils.logging_utils import Log


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


def test_generate_random_mask(height, width, density=0.5):
    """
    Generate a random boolean mask array.

    Args:
        height (int): Height of the mask array.
        width (int): Width of the mask array.
        density (float): Density of True values in the mask array (between 0 and 1).

    Returns:
        np.ndarray: Random boolean mask array of shape (height, width).
    """
    # Calculate the number of True values based on the density
    num_true = int(height * width * density)

    # Generate a random boolean mask array with specified density of True values
    mask = np.full((height, width), False, dtype=bool)
    indices = np.random.choice(height * width, size=num_true, replace=False)
    mask.flat[indices] = True

    return mask


def generate_segmentation_map(
    segmentation_masks_labels: torch.Tensor,
    color_palette: dict,
    mask: np.ndarray = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Generate a colorful segmentation map.

    Args:
        segmentation_masks_labels (torch.Tensor): Input tensor of shape (height, width).
            Tensor containing the predicted label for each pixel.
        color_palette (dict): A mapping of label index to corresponding RGB colors. For example:
            {
                '0': (0, 0, 0),        # Class 0 (Unknown class) represented as black (semantic background)
                '1': (255, 0, 0),      # Class 1 represented as red
                ...
            }
        mask (np.ndarray, optional): An optional mask array of shape (image_height, image_width)
            where True indicates pixels to be displayed and False indicates pixels to be masked out.

    Returns:
        np.ndarray: Segmentation map representing the colored pixel-wise predictions.
            Array of shape (image_height, image_width, 3) with RGB color values.
    """

    if debug:
        try:
            generate_segmentation_map.idx += 1
        except AttributeError:
            generate_segmentation_map.idx = 0

    # Convert tensor to numpy array and move it to CPU
    segmentation_masks_labels = (
        segmentation_masks_labels.detach().contiguous().cpu().numpy()
    )

    if debug:
        if generate_segmentation_map.idx == 0:
            print(
                f"segmentation_masks_labels.shape = {segmentation_masks_labels.shape}"
            )

    # Create an array of colors corresponding to class indices
    colors = np.array(
        [color_palette.get(label, (0, 0, 0)) for label in range(len(color_palette))],
        dtype=np.uint8,
    )

    if debug:
        if generate_segmentation_map.idx == 0:
            print(f"colors.shape = {colors.shape}")

    # Initialize segmentation map with default color (black, background color) for masked pixels
    segmentation_map = np.zeros(
        (segmentation_masks_labels.shape[0], segmentation_masks_labels.shape[1], 3),
        dtype=np.uint8,
    )

    if debug:
        if generate_segmentation_map.idx == 0:
            print(f"segmentation_map.shape = {segmentation_map.shape}")

    # Apply colors to pixels based on class indices, respecting the mask
    if mask is None:
        try:
            segmentation_map = colors[segmentation_masks_labels]
        except IndexError:
            segmentation_map = colors[np.zeros_like(segmentation_masks_labels)]
    else:
        try:
            segmentation_map[mask] = colors[segmentation_masks_labels[mask]]
        except IndexError:
            segmentation_map[mask] = colors[
                np.zeros_like(segmentation_masks_labels)[mask]
            ]

    if debug:
        if generate_segmentation_map.idx == 0:
            print(f"segmentation_map.shape = {segmentation_map.shape}")

    return segmentation_map


def generate_confidence_map(
    segmentation_masks_probabilities: torch.Tensor,
    color_map_name: str = "Spectral",
    debug: bool = False,
) -> np.ndarray:
    """
    Generate confidence map based on the softmax of segmentation logits (decoded semantics) tensor.

    Args:
        segmentation_masks_probabilities (torch.Tensor): Input tensor of shape (height * width).
            Tensor containing the pixel-wise predicted probability of the inferred class label.
        color_map_name (str): The name for matplotlib.colors.Colormap.
            Specifies the colormap to use for visualizing the confidence values.
            For available colormaps, refer to:
               [1] https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.get_cmap
               [2] https://matplotlib.org/stable/users/explain/colors/colormaps.html
        debug (bool, optional): If True, prints and saves data for further exploration.

    Returns:
        np.ndarray: Confidence map representing the confidence scores for each pixel in the input image.
            Array of shape (image_height, image_width, 3) with RGB color values.
    """

    if debug:
        try:
            generate_confidence_map.idx += 1
        except AttributeError:
            generate_confidence_map.idx = 0

    segmentation_masks_probabilities = (
        segmentation_masks_probabilities.detach().contiguous().cpu().numpy()
    )

    try:
        # Generate confidence map without alpha channel
        confidence_map = (
            generate_confidence_map.color_map(segmentation_masks_probabilities)[
                :, :, :3
            ]
            * 255
        ).astype(np.uint8)
    except AttributeError:
        generate_confidence_map.color_map = plt.cm.get_cmap(color_map_name)
        confidence_map = (
            generate_confidence_map.color_map(segmentation_masks_probabilities)[
                :, :, :3
            ]
            * 255
        ).astype(np.uint8)

    if debug:
        print(generate_confidence_map.idx)
        if generate_confidence_map.idx == 50:
            np.save("/tmp/max_probs_map.pt", segmentation_masks_probabilities)
            print(
                f"Save max_probs_map of {generate_confidence_map.idx}th frame to /tmp/max_probs_map.pt"
            )

    return confidence_map


def generate_semantic_mask(
    segmentation_masks_probabilities: torch.Tensor,
    threshold: float = 0.999,
    debug: bool = False,
) -> np.ndarray:
    """
    Generate a pixel-wise boolean mask to distinguish uncertain (False) and certain (True) segmentations based on probabilities.

    Args:
        segmentation_masks_probabilities: Input tensor of shape (image_height, image_width),
            containing the predicted probabilities (valued from 0 to 1)
            for the maximum probabilities among classes for each pixel.
        threshold: From 0 to 1, the boundary of uncertainty and certainty
        debug: Whether to print

    Returns:
        np.ndarray: A boolean array of shape (image_height, image_width)
    """

    if debug:
        try:
            generate_semantic_mask.idx += 1
        except AttributeError:
            generate_semantic_mask.idx = 0

    if debug:
        if generate_semantic_mask.idx == 0:
            print(
                f"segmentation_masks_probabilities.shape = {segmentation_masks_probabilities.shape}"
            )

    segmentation_masks_probabilities = (
        segmentation_masks_probabilities.detach().contiguous().cpu().numpy()
    )

    if debug:
        if generate_semantic_mask.idx == 0:
            print(
                f"segmentation_masks_probabilities.shape = {segmentation_masks_probabilities.shape}"
            )

    mask = segmentation_masks_probabilities >= threshold

    if debug:
        if generate_semantic_mask.idx == 0:
            print(f"mask.shape = {mask.shape}")

    return mask


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

        self.linear_layer = nn.Linear(
            self.semantic_embedding_dim, self.num_classes, bias=True
        )

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

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (semantic_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (1, num_classes, height * width),
                          representing pixel-wise predictions.
        """

        # Permute input tensor to shape (height, width, channels)
        x = x.permute(1, 2, 0)

        # Add a batch dimension
        x = x.unsqueeze(dim=0)

        try:
            # Reshape input tensor to (height * width, channels)
            x = x.view(-1, self.semantic_embedding_dim)
        except RuntimeError:
            Log(f"semantic_embedding_dim = {self.semantic_embedding_dim}", tag="Error")
            Log(
                "Check NUM_SEMANTIC_CHANNELS (C++ Macro) in 'submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h'",
                tag="Error",
            )
            raise

        # Pass through the linear layer to (height * width, num_classes)
        x = self.linear_layer(x)

        # Add a batch dimension, (1, height * width, num_classes)
        x = x.unsqueeze(dim=0)

        # (1, num_classes, height * width)
        x = x.permute(0, 2, 1)

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
