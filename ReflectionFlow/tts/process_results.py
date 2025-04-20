"""
Thanks ChatGPT for pairing.
"""

import glob
import argparse
import re
import os
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import make_image_grid  # Assuming this is available


def add_text_to_image(image: Image.Image, text: str, position=(10, 10), color="ivory") -> Image.Image:
    """
    Draws the given text on the image at the specified position.
    """
    draw = ImageDraw.Draw(image)
    try:
        # Try using a commonly available TrueType font with the specified font size.
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except Exception as e:
        print(f"Could not load TrueType font: {e}. Falling back to default font.")
        font = ImageFont.load_default()
    draw.text(position, text, fill=color, font=font)
    return image


def derive_collage_filename(prompt: str, sorted_filenames: list) -> str:
    """
    Derives a representative filename for the collage based on the group prompt and the range of i values.
    """
    # Use a regex to extract the i value from a filename (assuming the pattern _i@<number>_)
    i_pattern = re.compile(r"_i@(\d+)_")
    i_values = []
    for fname in sorted_filenames:
        m = i_pattern.search(fname)
        if m:
            i_values.append(int(m.group(1)))
    if not i_values:
        return f"collage_{prompt}.png"
    min_i, max_i = min(i_values), max(i_values)
    # Create a filename that shows the prompt and the i range.
    return f"collage_{prompt}_i@{min_i}-{max_i}.png"


def main(args):
    # Get all JSON files in the current directory that include 'hash' in their name.
    json_files = glob.glob(f"{args.path}/*.json")
    json_files = [f for f in json_files if "hash" in f]
    assert json_files

    # Regular expression to extract prompt, hash, i, and seed.
    pattern = re.compile(r"prompt@(.+?)_hash@([^_]+)_i@(\d+)_s@(\d+)\.json")

    # Group files by their prompt.
    groups = {}
    for filename in json_files:
        match = pattern.search(filename)
        if match:
            prompt, file_hash, i_str, seed = match.groups()
            groups.setdefault(prompt, []).append(filename)

    print(f"Total groups found: {len(groups)}.")

    # Process each group separately.
    for prompt, files in groups.items():
        # Sort filenames in the group by the integer value of i i.e., the search.
        sorted_files = sorted(files, key=lambda fname: int(pattern.search(fname).group(3)))

        # Load corresponding PNG images and annotate them with the i value.
        annotated_images = []
        for fname in sorted_files:
            # Extract the i value from the filename.
            i_val = int(pattern.search(fname).group(3))
            # Replace .json with .png to get the image filename.
            png_filename = fname.replace(".json", ".png")
            try:
                image = Image.open(png_filename)
            except Exception as e:
                print(f"Could not open image '{png_filename}': {e}")
                continue

            # Annotate the image with "i=<value>" in the top-left corner.
            annotated_image = add_text_to_image(image, f"i={i_val}")
            annotated_images.append(annotated_image)

        if not annotated_images:
            print(f"No valid images for prompt '{prompt}'.")
            continue

        # Create a collage (horizontal grid: one row, all images as columns).
        grid = make_image_grid(annotated_images, rows=1, cols=len(annotated_images))

        # Derive a representative collage filename.
        collage_filename = derive_collage_filename(prompt, sorted_files)
        collage_filename = os.path.join(args.path, collage_filename)
        grid.save(collage_filename)
        print(f"Saved collage for prompt '{prompt}' as '{collage_filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path containing the JSON AND image files.")
    args = parser.parse_args()
    main(args)
