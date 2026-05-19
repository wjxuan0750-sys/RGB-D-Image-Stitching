"""
Depth Stitching v12 -- Center the bowl, then blend
====================================================
Detects the bowl in each depth image via multi-scale template matching,
centers + scales every image so the bowl aligns, then blends by averaging.

Usage:
    python scripts/stitch_depth.py           # processes both scenes
    python scripts/stitch_depth.py scene1    # 2-image scene only
    python scripts/stitch_depth.py scene2    # 3-image scene only

Input:  data/depth/scene1/*.png   (2 images)
        data/depth/scene2/*.png   (3 images)
Output: results/stitched_depth_1.png
        results/stitched_depth_2.png
"""

import os, sys, cv2, numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA_BASE = os.path.join(ROOT, "data", "depth")
OUT_DIR = os.path.join(ROOT, "results")


def detect_bowl(depth_img):
    """Multi-scale circular template matching. Returns (cx, cy, r) or None."""
    h, w = depth_img.shape
    valid = depth_img > 0

    p2, p98 = np.percentile(depth_img[valid], 2), np.percentile(depth_img[valid], 98)
    dn = np.clip((depth_img - p2) / max(p98 - p2, 1), 0, 1).astype(np.float32)
    dn[depth_img == 0] = 0.5

    best_score, best_result = -np.inf, None
    min_r, max_r = max(30, min(w, h) // 30), min(700, min(w, h) // 3)

    for r in range(min_r, max_r, 15):
        ts = int(r * 2.5)
        if ts < 20 or ts > min(w, h):
            continue
        y, x = np.ogrid[-ts // 2:ts // 2, -ts // 2:ts // 2]
        dist = np.sqrt(x.astype(np.float32) ** 2 + y.astype(np.float32) ** 2)

        for polarity in [-1, 1]:
            tmpl = np.zeros((ts, ts), dtype=np.float32)
            tmpl[dist < r * 0.75] = -1.0 * polarity
            tr = (dist >= r * 0.75) & (dist < r * 1.25)
            frac = np.clip((dist[tr] - r * 0.75) / (r * 0.5), 0, 1)
            tmpl[tr] = (2.0 * frac - 1.0) * polarity
            tmpl -= np.mean(tmpl)
            tn = np.sqrt(np.sum(tmpl ** 2))
            if tn < 1e-6:
                continue
            tmpl /= tn

            try:
                corr = cv2.matchTemplate(dn, tmpl, cv2.TM_CCOEFF_NORMED)
                _, maxVal, _, maxLoc = cv2.minMaxLoc(corr)
                if maxVal > best_score:
                    best_score = maxVal
                    best_result = (maxLoc[0] + ts // 2, maxLoc[1] + ts // 2, r)
            except Exception:
                continue

    return best_result if (best_result and best_score > 0.15) else None


def center_transform(bowl, target_cx, target_cy, target_r):
    """3x3 matrix mapping bowl to (target_cx, target_cy) at radius target_r."""
    cx, cy, r = bowl
    s = target_r / max(r, 1e-6)
    return np.array([[s, 0, target_cx - s * cx],
                     [0, s, target_cy - s * cy],
                     [0, 0, 1]], dtype=np.float64)


def blend_simple(imgs, transforms, cw, ch):
    """Average blending with per-pixel valid-source masking."""
    accum = np.zeros((ch, cw), dtype=np.float64)
    counts = np.zeros((ch, cw), dtype=np.float64)
    for img, H in zip(imgs, transforms):
        warped = cv2.warpPerspective(img.astype(np.float32), H, (cw, ch),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        src_mask = (img > 0).astype(np.uint8)
        warped_mask = cv2.warpPerspective(src_mask, H, (cw, ch),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ok = (warped_mask > 0) & (warped > 0)
        accum[ok] += warped[ok]
        counts[ok] += 1.0
    valid = counts > 0
    accum[valid] /= counts[valid]
    return accum


def crop_black(img):
    """Trim dark borders iteratively."""
    gray = img.astype(np.float32)
    gmin, gmax = np.min(gray), np.max(gray)
    if gmax <= gmin:
        return img
    gn = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)
    for _ in range(200):
        rows = np.any(gn > 5, axis=1)
        cols = np.any(gn > 5, axis=0)
        if not np.any(rows) or not np.any(cols):
            return img[:gn.shape[0], :gn.shape[1]]
        t = np.argmax(rows)
        b = len(rows) - np.argmax(rows[::-1])
        l = np.argmax(cols)
        r = len(cols) - np.argmax(cols[::-1])
        changed = False
        if t > 0 and np.sum(gn[t, l:r] < 5) / max(r - l, 1) > 0.03:
            gn = gn[t + 1:, :]; changed = True
        if b < gn.shape[0] and np.sum(gn[b - 1, l:r] < 5) / max(r - l, 1) > 0.03:
            gn = gn[:b - 1, :]; changed = True
        if l > 0 and np.sum(gn[t:b, l] < 5) / max(b - t, 1) > 0.03:
            gn = gn[:, l + 1:]; changed = True
        if r < gn.shape[1] and np.sum(gn[t:b, r - 1] < 5) / max(b - t, 1) > 0.03:
            gn = gn[:, :r - 1]; changed = True
        if not changed:
            break
    return img[:gn.shape[0], :gn.shape[1]]


def stitch_scene(scene_name):
    data_dir = os.path.join(DATA_BASE, scene_name)
    depth_files = sorted(glob(os.path.join(data_dir, "*.png")))
    n = len(depth_files)
    if n < 2:
        print(f"  Need >= 2 images, found {n}"); return

    depths, orig_h, orig_w = [], None, None
    for f in depth_files:
        d = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if d.ndim == 3: d = d[:, :, 0]
        depths.append(d.astype(np.float32))
        if orig_h is None:
            orig_h, orig_w = d.shape[:2]

    dtype0 = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED).dtype
    print(f"  {n} images, {orig_w}x{orig_h}, dtype={dtype0}")

    bowls = [detect_bowl(d) for d in depths]
    for i, (f, b) in enumerate(zip(depth_files, bowls)):
        tag = f"({b[0]:.0f},{b[1]:.0f}) r={b[2]:.0f}" if b else "FAILED"
        print(f"  bowl {os.path.basename(f)}: {tag}")
    if any(b is None for b in bowls):
        print("  ERROR: bowl detection failed"); return

    ref_idx = int(np.argmax([b[2] for b in bowls]))
    ref = bowls[ref_idx]
    print(f"  reference: img{ref_idx + 1}  r={ref[2]:.0f}")

    margin = 300
    canvas_w, canvas_h = orig_w + margin * 2, orig_h + margin * 2
    target_cx, target_cy = canvas_w / 2.0, canvas_h / 2.0

    transforms = [center_transform(b, target_cx, target_cy, ref[2]) for b in bowls]
    for i, H in enumerate(transforms):
        if i != ref_idx:
            print(f"  img{i + 1}: s={H[0,0]:.3f}  dx={H[0,2]:.0f}  dy={H[1,2]:.0f}")

    result = blend_simple(depths, transforms, canvas_w, canvas_h)
    result = crop_black(result)

    if np.issubdtype(dtype0, np.floating):
        out = result.astype(dtype0)
    else:
        out = np.round(result).clip(0, np.iinfo(dtype0).max).astype(dtype0)

    scene_num = scene_name.replace("scene", "")
    out_path = os.path.join(OUT_DIR, f"stitched_depth_{scene_num}.png")
    cv2.imwrite(out_path, out)
    print(f"  -> {out_path}  ({out.shape[1]}x{out.shape[0]})\n")


# ================================================================
if __name__ == "__main__":
    import sys
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["scene1", "scene2"]
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Depth Stitching v12 -- center bowl, then blend")
    print("=" * 60)
    for s in scenes:
        print(f"\n--- {s} ---")
        stitch_scene(s)
    print("Done.")
