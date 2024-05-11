import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    confidence_map = np.load(
        "/mnt/dev-ssd-8T/shuqixiao/dev/projects/MonoGS/datasets/temp/max_probs_map.npy"
    )

    # Flatten the confidence map
    flat_image = confidence_map.flatten()

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the confidence map
    axs[0].imshow(confidence_map, cmap="gray")
    axs[0].set_title("Confidence Map")

    # Plot the histogram
    hist, bins, _ = axs[1].hist(flat_image, bins=256, range=(0.95, 1), color="gray", alpha=0.7)
    axs[1].set_xlabel("Confidence")
    axs[1].set_ylabel("Number of Pixels")
    axs[1].set_title("Confidence Histogram")
    axs[1].set_yscale('log')  # Set y-axis scale to logarithmic

    # Apply binary thresholding
    binary_mask = confidence_map >= 0.9995

    # Plot the binary mask
    axs[2].imshow(binary_mask, cmap="gray")
    axs[2].set_title("Binary Mask")

    # Show the plots
    plt.show()
