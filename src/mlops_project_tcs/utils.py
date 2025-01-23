import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from PIL import Image
from typing import List, Tuple, Optional, Union
from pathlib import Path


def plot_image_grid_with_labels(
    image_paths: List[Union[str, Path]],
    rows: int,
    cols: int,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    titles: Optional[List[str]] = None,
    show: bool = False,
) -> None:
    """
    Plots a grid of images with optional titles and labels.

    Args:
        image_paths (list of str or Path): List of image file paths.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        labels (list, optional): Labels for each image. Should have the same length as images.
        figsize (tuple): Size of the entire figure (default is (10, 10)).
        titles (list, optional): Titles for each image. Should have the same length as images.
        show (bool): Will run plt.show() if true.
    """
    if len(image_paths) < rows * cols:
        raise ValueError("Not enough images to fill the grid.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of a 2D array of axes

    for i, ax in enumerate(axes):
        # Load the image
        img = Image.open(image_paths[i]).convert("RGB")
        img = np.array(img)

        # Normalize if necessary
        if img.max() > 1:
            img = img / 255.0  # Normalize to [0, 1] range if in [0, 255]

        # Display the image
        ax.imshow(img)
        ax.axis("off")  # Hide axes

        # Add title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)

        # Add label below the image
        if labels and i < len(labels):
            ax.text(0.5, -0.1, labels[i], fontsize=9, ha="center", transform=ax.transAxes)  # Positioned below the image

    plt.tight_layout()

    if show:
        plt.show()


def get_targets_from_subset(subset: Subset) -> torch.Tensor:
    """Extract targets from a Subset."""
    targets = []
    for _, label in subset:
        targets.append(label)
    return torch.tensor(targets)
