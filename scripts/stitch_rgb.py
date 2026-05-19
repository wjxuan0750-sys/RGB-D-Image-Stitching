"""
RGB Image Stitching v21 -- Yellow-bowl detection + similarity alignment
========================================================================
Detects the yellow bowl in each RGB image (HSV color space),
aligns via translate+scale transforms, blends by averaging.

Usage:
    python scripts/stitch_rgb.py           # processes both scenes
    python scripts/stitch_rgb.py scene1    # 2-image scene only
    python scripts/stitch_rgb.py scene2    # 2-image scene only

Input:  data/rgb/scene1/*.bmp   (2 images)
        data/rgb/scene2/*.bmp   (2 images)
Output: results/stitched_rgb_1.bmp
        results/stitched_rgb_2.bmp
"""

import os, sys, cv2, numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_BASE = os.path.join(ROOT, "data", "rgb")
OUT_DIR = os.path.join(ROOT, "results")

YELLOW_LOW = np.array([12, 60, 60])
YELLOW_HIGH = np.array([38, 255, 255])


def detect_bowl(img):
    """Detect yellow bowl in RGB via HSV. Returns (cx, cy, r) or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return None
    hull = cv2.convexHull(c)
    for src in (c, hull):
        if len(src) < 5:
            continue
        try:
            el = cv2.fitEllipse(src)
            if min(el[1]) / max(el[1]) >= 0.15:
                (cx, cy), (a0, a1), _ = el
                return (cx, cy, np.sqrt(max(a0, a1) * min(a0, a1)))
        except Exception:
            continue
    return None


def align(ref, new):
    """Similarity transform from 'new' bowl to 'ref' bowl."""
    cx_r, cy_r, r_r = ref
    cx_n, cy_n, r_n = new
    s = r_r / max(r_n, 1e-6)
    return np.array([[s, 0, cx_r - s * cx_n],
                     [0, s, cy_r - s * cy_n],
                     [0, 0, 1]], dtype=np.float64)


def crop_black(img):
    """Trim edges until each edge has < 5% black pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top, bottom = 0, img.shape[0]
    left, right = 0, img.shape[1]
    for _ in range(200):
        changed = False
        if top < bottom:
            row = gray[top, left:right]
            if np.sum(row < 5) / max(len(row), 1) > 0.05:
                top += 1; changed = True
        if bottom > top:
            row = gray[bottom - 1, left:right]
            if np.sum(row < 5) / max(len(row), 1) > 0.05:
                bottom -= 1; changed = True
        if left < right:
            col = gray[top:bottom, left]
            if np.sum(col < 5) / max(len(col), 1) > 0.05:
                left += 1; changed = True
        if right > left:
            col = gray[top:bottom, right - 1]
            if np.sum(col < 5) / max(len(col), 1) > 0.05:
                right -= 1; changed = True
        if not changed:
            break
    return img[top:bottom, left:right]


def stitch_scene(scene_name):
    data_dir = os.path.join(DATA_BASE, scene_name)
    files = sorted(glob(os.path.join(data_dir, "*.bmp")))
    if len(files) < 2:
        print(f"  Need >= 2 images, found {len(files)}"); return

    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  Cannot read: {f}"); return
        imgs.append(img)
        print(f"  {os.path.basename(f)}: {img.shape[1]}x{img.shape[0]}")

    bowls = [detect_bowl(im) for im in imgs]
    for i, (f, b) in enumerate(zip(files, bowls)):
        tag = f"({b[0]:.0f},{b[1]:.0f}) r={b[2]:.0f}" if b else "FAILED"
        print(f"  bowl {os.path.basename(f)}: {tag}")
    if any(b is None for b in bowls):
        print("  Bowl detection failed!"); return

    best = max(range(len(bowls)), key=lambda i: bowls[i][2])
    ref_bowl = bowls[best]
    ref_img = imgs[best]
    print(f"  reference: img{best + 1}  r={ref_bowl[2]:.0f}")

    # Estimate canvas
    all_corners = []
    transforms = {best: np.eye(3, dtype=np.float64)}
    for i, (im, bowl) in enumerate(zip(imgs, bowls)):
        if i == best:
            continue
        H = align(ref_bowl, bowl)
        transforms[i] = H
        h, w = im.shape[:2]
        cn = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        cw = cv2.perspectiveTransform(cn, H)
        all_corners.append(cw.reshape(-1, 2))
    h, w = ref_img.shape[:2]
    all_corners.append(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1],
                                    [0, h - 1]]))
    all_c = np.vstack(all_corners)
    [xmin, ymin] = np.int32(all_c.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_c.max(axis=0).ravel())
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    cw, ch = xmax - xmin + 1, ymax - ymin + 1
    print(f"  canvas: {cw}x{ch}")

    # Blend
    accum = np.zeros((ch, cw, 3), dtype=np.float64)
    hits = np.zeros((ch, cw), dtype=np.float64)
    for i, (im, bowl) in enumerate(zip(imgs, bowls)):
        H = transforms[i]
        warped = cv2.warpPerspective(im, T @ H, (cw, ch))
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 3
        accum[mask] += warped[mask].astype(np.float64)
        hits[mask] += 1.0
    valid = hits > 0
    accum[valid] /= hits[valid, np.newaxis]
    result = np.clip(accum, 0, 255).astype(np.uint8)
    result = crop_black(result)

    scene_num = scene_name.replace("scene", "")
    out_path = os.path.join(OUT_DIR, f"stitched_rgb_{scene_num}.bmp")
    cv2.imwrite(out_path, result)
    print(f"  -> {out_path}  ({result.shape[1]}x{result.shape[0]})\n")


# ================================================================
if __name__ == "__main__":
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["scene1", "scene2"]
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("RGB Stitching v21 -- yellow-bowl alignment")
    print("=" * 60)
    for s in scenes:
        print(f"\n--- {s} ---")
        stitch_scene(s)
    print("Done.")
