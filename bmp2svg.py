import numpy as np
import skimage as ski
from scipy import ndimage
import svgwrite

def optimize_seq(seq):
    """
    Given a list of x,y pairs, remove all unnecessary points (that lie along a straight line
    between two other points.
    """
    i = 1
    while i + 1 < len(seq):
        if any(seq[i][xy] == seq[i-1][xy] and seq[i][xy] == seq[i+1][xy] for xy in [0, 1]):
            del seq[i]
        else:
            i += 1
    if seq[0] == seq[-1] and len(seq) > 1:
        return seq[:-1]
    else:
        return seq

def trace_path(edges, mask):
    """
    Given an NxMx4 boolean array of edges and masked region to consider, this will
    iterateivly walk along the edges of the shape, stringing them together into
    a single SVG path
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    edges = edges.copy()
    path = ""

    # Find all pixels in the masked area that are also on an edge
    mask = np.logical_and(np.any(edges, axis=2), mask)

    # Repeat whole process as long as edges are left (in case of shapes with enclosed holes in the middle)
    while np.any(mask):
        edges[~mask] = 0
        indices = np.argwhere(mask)

        # Add first point to path
        start_index = indices[0]
        seq = []
        if edges[*start_index, UP]:
            start_edge = UP
            seq.append(start_index)
        elif edges[*start_index, DOWN]:
            start_edge = DOWN
            seq.append(start_index + [1, 1])
        elif edges[*start_index, RIGHT]:
            start_edge = RIGHT
            seq.append(start_index + [0, 1])
        elif edges[*start_index, LEFT]:
            start_edge = LEFT
            seq.append(start_index + [1, 0])
        
        cursor = (start_index, start_edge)

        # Coming from a previous edge in a given direction, there are only a few other
        # pixel + edge direction combinations that could connect to it. Only look clockwise
        # (i.e. from the "right" point of the edge) to keep things simpler
        connections = {
            UP: [
                ([ 1,  0], UP),
                ([ 1, -1], LEFT),
                ([ 0,  0], RIGHT),
            ],
            DOWN: [
                ([-1,  0], DOWN),
                ([-1,  1], RIGHT),
                ([ 0,  0], LEFT),
            ],
            RIGHT: [
                ([ 0,  1], RIGHT),
                ([ 1,  1], UP),
                ([ 0,  0], DOWN),
            ],
            LEFT: [
                ([ 0, -1], LEFT),
                ([-1, -1], DOWN),
                ([ 0,  0], UP),
            ],
        }

        # For each pixel edge, look at all possible edges it could connect to. If there are none,
        # That means it must have connected back to the start
        while True:
            p, edge = cursor
            # Delete edges we have already handled so we don't go in circles
            edges[*p, edge] = False
            if edge == UP:
                seq.append(p + [0, 1])
            elif edge == DOWN:
                seq.append(p + [1, 0])
            elif edge == LEFT:
                seq.append(p + [0, 0])
            elif edge == RIGHT:
                seq.append(p + [1, 1])
            done = True
            for [dx, dy], new_edge in connections[edge]:
                delta = [dy, dx]
                if edges[*(p + delta), new_edge]:
                    cursor = (p + delta, new_edge)
                    done = False
                    break
            if done:
                break

        # switch from y,x to x,y coordinates and prune sequence
        seq = optimize_seq([tuple(reversed(p)) for p in seq])
        # Build up SVG path from list of points
        path += f"M {seq[0][0]},{seq[0][1]} "
        path += " ".join(f"L {x},{y}" for x, y in seq[1:])
        path += " z "
        # Update our mask, removing all already handled edges, in case we need to go again
        mask = np.logical_and(np.any(edges, axis=2), mask)
    return path[:-1]


def bmp2svg(input_file, output_file, bg_color):
    img = ski.io.imread(input_file)[:, :, :3]
    img = ski.util.img_as_ubyte(img)
    
    # Convert numti-channel image to single channel. TODO handle grayscale images and alpha channel
    img_rgb = np.left_shift(img[:,:, 0].astype(np.uint32), 16) + np.left_shift(img[:,:, 1].astype(np.uint32), 8) + img[:,:, 2].astype(np.uint32)
    bg_color_rgb = np.left_shift(bg_color[0], 16) + np.left_shift(bg_color[1], 8) + bg_color[2]

    # Flood-fill to find regions of the same color
    labels, num_labels = ski.measure.label(img_rgb, return_num=True, connectivity=1, background=bg_color_rgb)

    # Edge-detection filters
    kernel_top = np.flip(np.array(
        [[0, -1, 0],
         [0, 1, 0],
         [0, 0, 0]], dtype=np.int8))
    kernel_bottom = np.flip(np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, -1, 0]], dtype=np.int8))
    kernel_left = np.flip(np.array(
        [[0, 0, 0],
         [-1, 1, 0],
         [0, 0, 0]], dtype=np.int8))
    kernel_right = np.flip(np.array(
        [[0, 0, 0],
         [0, 1, -1],
         [0, 0, 0]], dtype=np.int8))

    # Create an NxMx4 boolean array where each cell represents if that pixel is at the
    # edge of the shape it's in looking in each of the four directions
    edges = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=bool)
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    for label in range(1, num_labels+1):
        mask = labels == label
        top_edges = ndimage.convolve(mask.astype(np.int8), kernel_top, mode="constant", cval=0) == 1
        bottom_edges = ndimage.convolve(mask.astype(np.int8), kernel_bottom, mode="constant", cval=0) == 1
        left_edges = ndimage.convolve(mask.astype(np.int8), kernel_left, mode="constant", cval=0) == 1
        right_edges = ndimage.convolve(mask.astype(np.int8), kernel_right, mode="constant", cval=0) == 1
        edges[top_edges, UP] = True
        edges[bottom_edges, DOWN] = True
        edges[left_edges, LEFT] = True
        edges[right_edges, RIGHT] = True

    # Trace the border in each labelled region. Save the SVG path and color to a list
    paths = []
    for label in range(1, num_labels+1):
        mask = labels==label
        data = trace_path(edges, mask)
        color = img[mask][0]
        paths.append((color, data))

    # Create SVG
    height, width, _ = img.shape
    svg = svgwrite.Drawing(output_file, profile='tiny')
    svg.viewbox(width=width, height=height)
    svg.add(svg.rect(size=(width, height), fill=f"rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})"))
    for color, data in paths:
        svg.add(svg.path(d=data, fill=f"rgb({color[0]},{color[1]},{color[2]})"))
    svg.save()


if __name__ == "__main__":
    import argparse
    import re
    import os
    import sys

    parser = argparse.ArgumentParser(description="Converts a pixel art raster image to an SVG by drawing regions of the same color as SVG paths")
    parser.add_argument("input_file", metavar="INPUT_FILE", help="Raster image in any format")
    parser.add_argument("-bg", metavar="HEX", default="#000000", help="Color to treat as the background")
    parser.add_argument("-o", metavar="OUTPUT_FILE", dest="output_file", help="SVG file to write output to")


    args = parser.parse_args()

    match = re.match("#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", args.bg)
    if match is None:
        print(f"Invalid HEX format for bg color '{args.bg}'")
        sys.exit(1)
    bg_color = [int(n) for n in match.groups()]

    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f"Invalid input file '{input_file}'")
        sys.exit(1)

    output_file = args.output_file
    if not output_file:
        fname, _ = os.path.splitext(input_file)
        output_file = f"{fname}.svg"

    bmp2svg(input_file, output_file, bg_color)

    