import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    uncertainty_map = np.load(
        "/mnt/dev-ssd-8T/shuqixiao/dev/projects/MonoGS/datasets/temp/max_probs_map.npy"
    )

    # Flatten the uncertainty map
    flat_image = uncertainty_map.flatten()

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the uncertainty map
    axs[0].imshow(uncertainty_map, cmap="gray")
    axs[0].set_title("Certainty Map")

    # Plot the histogram
    # TODO: Histogram equalization
    hist, bins, _ = axs[1].hist(
        flat_image, bins=256, range=(0.95, 1), color="gray", alpha=0.7
    )
    axs[1].set_xlabel("Certainty")
    axs[1].set_ylabel("Number of Pixels")
    axs[1].set_title("Certainty Histogram")
    axs[1].set_yscale("log")  # Set y-axis scale to logarithmic
    axs[1].set_title("Certainty Histogram")
    axs[1].set_yscale("log")  # Set y-axis scale to logarithmic

    # Apply binary thresholding
    binary_mask = uncertainty_map >= 0.9995

    # Plot the binary mask
    axs[2].imshow(binary_mask, cmap="gray")
    axs[2].set_title("Binary Mask")

    # Show the plots
    plt.show()
