# RGB-D Image Stitching

**Bowl-based alignment for RGB and depth image stitching.**

Given 2–3 images of a scene containing a bowl (captured from different viewpoints), the scripts detect the bowl, align the images via similarity transforms, and blend them into a seamless stitched result.

## File Structure

```
RGB-D-Image-Stitching/
├── scripts/
│   ├── stitch_depth.py      # Depth image stitching (template-matching bowl detection)
│   └── stitch_rgb.py        # RGB image stitching (HSV yellow-bowl detection)
├── data/
│   ├── depth/
│   │   ├── scene1/          # 2 depth images
│   │   └── scene2/          # 3 depth images
│   └── rgb/
│       ├── scene1/          # 2 RGB images
│       └── scene2/          # 2 RGB images
├── results/                 # Stitched outputs
│   ├── stitched_depth_1.png
│   ├── stitched_depth_2.png
│   ├── stitched_rgb_1.bmp
│   └── stitched_rgb_2.bmp
└── README.md
```

## Usage

```bash
# Install dependencies
pip install opencv-python numpy

# Run both scenes (depth)
python scripts/stitch_depth.py

# Run both scenes (RGB)
python scripts/stitch_rgb.py

# Run a single scene
python scripts/stitch_depth.py scene1
python scripts/stitch_rgb.py scene2
```

## Method

### Depth Stitching
- **Bowl detection:** Multi-scale normalized cross-correlation with a synthetic circular bowl template (dark interior + bright rim)
- **Alignment:** Translate + scale each image so the bowl centers at the same position with uniform radius
- **Blending:** Pixel-wise average over overlapping valid regions, with source-mask propagation to prevent edge artifacts

### RGB Stitching
- **Bowl detection:** HSV color-space thresholding for yellow bowl, followed by ellipse fitting on the largest contour
- **Alignment:** Similarity transform (scale + translate) relative to the largest detected bowl
- **Blending:** Simple averaging with binary validity masks

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
