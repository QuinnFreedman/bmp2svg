A tool to convert bitmap images (e.g. PNGs), especially low-resolution pixel art, to infinitely scalable SVGs for displaying on the web. This is done more intelligently than going pixel-by-pixel or row-by-row. Each solid block of color is converted to a single SVG path with a single rectangle for the background, resulting in a small and performant file.

Usage:

```
python3 ./bmp2svg.py input_file.png -o output_file.svg -bg #000000
```
