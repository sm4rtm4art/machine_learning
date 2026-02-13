#!/usr/bin/env python
"""
    Tool for manual labeling
    - loads images from a given path
    - depicts images one-by-one and waits for keyboard input
    - feature present? 1=true, 0=false
    - no storing yet
"""

from pathlib import Path

import typer
from ml_portfolio.common.paths import get_project_paths

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

app = typer.Typer()

PROJECT_NAME = "vision_ssl_transfer"





"""  
    @param: input_dir: str, default is set to Oxford Pets default download path
    @param: shuffle: bool, default is true, shuffle images or not
    !! use typer.Options in next version
"""
@app.command()
def main(
    input_dir = get_project_paths(PROJECT_NAME).data_dir / "images",
    shuffle = True
) -> dict[str, bool]:

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {input_dir}")

    image_paths = sorted(input_dir.glob("*.jpg"))


    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {input_dir}")

    # found JPGs in input_dir, run labeling

    if shuffle:
        random.shuffle(image_paths)

    def set_title(ax, idx, image_count):
        ax.set_title(f"Press 0 or 1 (Image {idx + 1}/{image_count})\n Press q to quit")

    labels = {}
    current_idx = 0

    # Load the first image
    image = mpimg.imread(image_paths[current_idx])

    # Create figure once
    fig, ax = plt.subplots()
    image_obj = ax.imshow(image)
    ax.axis("off")
    set_title(ax,current_idx,len(image_paths))

    def on_key(event):
        nonlocal current_idx
        key = event.key
        if key in ["0", "1"]:
            # Save label as boolean
            labels[str(image_paths[current_idx])] = key == "1"

            # Move to next image
            current_idx += 1
            if current_idx >= len(image_paths):
                plt.close(fig)  # all images labeled
                return

            # Update image in the same axes
            new_image = mpimg.imread(image_paths[current_idx])
            image_obj.set_data(new_image)
            set_title(ax,current_idx,len(image_paths))
            fig.canvas.draw()  # refresh figure

        elif key == "q":
            plt.close(fig)  # quit early

    # Connect callback
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # blocks until plt.close() is called
    print(labels)
    for key, value in labels.items():
        print(f"{key}: {value}")
    return labels



if __name__ == "__main__":
    app()

