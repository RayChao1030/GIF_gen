import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
from tqdm import tqdm
import imageio
import os
import tempfile
import shutil

def load_lg_file(path):
    """
    Reads a .lg file and extracts die dimensions, blocks, and placement rows.
    
    Args:
        path (str): The file path to the .lg file.
    
    Returns:
        tuple: A tuple containing:
            - die_data (tuple): (x1, y1, x2, y2) defining the die size.
            - block_data (list): List of blocks, each as (name, x, y, width, height, status).
            - row_data (list): List of rows, each as (start_x, start_y, site_width, site_height, total_sites).
    """
    def parse_die_size(tokens):
        """Parses the die size line."""
        return tuple(map(int, tokens[1:5]))

    def parse_row(tokens):
        """Parses a placement row line."""
        return tuple(map(int, tokens[1:6]))

    def parse_block(tokens):
        """Parses a block line."""
        name, x, y, width, height, status = tokens
        return name, int(x), int(y), int(width), int(height), status

    die_data = None
    block_data = []
    row_data = []

    try:
        with open(path, "r") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue  # Ignore comments and empty lines
                tokens = stripped.split()
                if stripped.startswith("DieSize") and len(tokens) == 5:
                    die_data = parse_die_size(tokens)
                elif stripped.startswith("PlacementRows") and len(tokens) == 6:
                    row_data.append(parse_row(tokens))
                elif len(tokens) == 6 and tokens[-1] in {"FIX", "NOTFIX"}:
                    block_data.append(parse_block(tokens))
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Error processing file: {e}")
    
    return die_data, block_data, row_data

def load_opt_file(path):
    """
    Reads an .opt file and extracts optimization steps.
    
    Args:
        path (str): File path to the .opt file.
    
    Returns:
        list: A list of dictionaries representing optimization steps.
    """
    def parse_line(line):
        """Helper to parse a single line into a structured step."""
        header, details = line.split("-->")
        cells_to_remove = header.replace("Banking_Cell:", "").split()
        block_info = details.split()
        new_block = {
            "name": block_info[0],
            "x": int(block_info[1]),
            "y": int(block_info[2]),
            "width": int(block_info[3]),
            "height": int(block_info[4]),
        }
        return {"remove": cells_to_remove, "add": new_block}

    try:
        with open(path, "r") as file:
            return [
                parse_line(line.strip())
                for line in file
                if line.startswith("Banking_Cell:")
            ]
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def load_post_file(path):
    """Parses .post file to extract post-processing results."""
    result_data = []
    with open(path, "r") as file:
        for line in file:
            content = line.strip()
            if not content or content.startswith("#"):
                continue
            tokens = content.split()
            if len(tokens) >= 2:
                result_data.append({
                    "position": (int(tokens[0]), int(tokens[1])),
                    "moved": []
                })
    return result_data


# Step 2: Visualization
def visualize_static_layout(ax, rows, die):
    """Draws static layout elements like rows and die area."""
    row_rectangles = []
    for row in rows:
        start_x, start_y, site_width, site_height, total_sites = row
        rectangle = patches.Rectangle(
            (start_x, start_y), site_width * total_sites, site_height,
            edgecolor="gray", linestyle="dashed", facecolor="none", linewidth=0.8
        )
        row_rectangles.append(rectangle)
    ax.add_collection(PatchCollection(row_rectangles, match_original=True))
    ax.set_xlim(die[0] - 10, die[2] + 10)
    ax.set_ylim(die[1] - 10, die[3] + 10)
    ax.set_aspect("equal")


def generate_animation(die, initial_blocks, rows, optimization_steps, post_results, output_file):
    """Creates an animation showing layout optimization progress."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
    #plt.style.use("seaborn-dark")
    frames = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        num_steps = len(optimization_steps)
        progress_steps = np.linspace(0, num_steps, min(30, num_steps), dtype=int)

        for step in tqdm(progress_steps, desc="Rendering frames"):
            ax.clear()
            visualize_static_layout(ax, rows, die)

            active_blocks = initial_blocks.copy()
            removed_blocks = set()
            merged_blocks = []

            for i in range(step):
                removed_blocks.update(optimization_steps[i]["remove"])
                new_cell = optimization_steps[i]["add"]
                merged_position = post_results[i]["position"]
                merged_blocks.append((
                    new_cell["name"], merged_position[0], merged_position[1],
                    new_cell["width"], new_cell["height"], "MERGED"
                ))

            active_blocks = [b for b in active_blocks if b[0] not in removed_blocks] + merged_blocks

            # Draw current blocks
            for block in active_blocks:
                rect = patches.Rectangle(
                    (block[1], block[2]), block[3], block[4],
                    color={"FIX": "#E74C3C", "NOTFIX": "#3498DB", "MERGED": "#FFFF44"}[block[5]],
                    alpha=0.8
                )
                ax.add_patch(rect)

            ax.set_title(f"Step {step}/{num_steps}", fontsize=14)
            temp_file = os.path.join(temp_dir, f"frame_{step:03d}.png")
            plt.savefig(temp_file, bbox_inches="tight")
            frames.append(imageio.imread(temp_file))

        imageio.mimsave(output_file, frames, fps=5)
        print(f"Animation saved to {output_file}")
    finally:
        shutil.rmtree(temp_dir)


# Step 3: Entry Point
def run_tool():
    """Handles argument parsing and calls necessary functions."""
    parser = argparse.ArgumentParser(description="Layout Optimization Animation Tool")
    parser.add_argument("-lg", required=True)
    parser.add_argument("-opt", required=True)
    parser.add_argument("-post", required=True)
    parser.add_argument("-output", default="output_animation.gif")
    args = parser.parse_args()

    try:
        print("Loading files...")
        die, blocks, rows = load_lg_file(args.lg)
        opt_steps = load_opt_file(args.opt)
        post_data = load_post_file(args.post)
        print("Generating animation...")
        generate_animation(die, blocks, rows, opt_steps, post_data, args.output)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_tool()