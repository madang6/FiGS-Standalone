"""Generate a checkerboard calibration pattern with a 2-square white border.

Run once. Display the saved PNG on a monitor at 100% zoom (or print at 100% scale),
then record calibration videos.

Usage (inside the FiGS Docker container, from repo root):
    python src/generate_checkerboard.py

The PNG is saved to <REPO_ROOT>/calibration_patterns/. Because the repo is
bind-mounted into the container, the file is immediately accessible on the host.
"""

from pathlib import Path
import os
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PATTERN = (9, 6)          # inner corners (cols, rows). Pattern is (cols+1) x (rows+1) squares.
SQUARE_PX = 80            # rendered size per square in pixels (visual only)
BORDER_SQUARES = 2        # white margin around the pattern, in square widths


# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------
def find_repo_root() -> Path:
    p = Path(os.getcwd()).resolve()
    for candidate in [p] + list(p.parents):
        if (candidate / "docker-compose.yml").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate FiGS-Standalone repo root from {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inner_cols, inner_rows = PATTERN
    pat_cols = inner_cols + 1
    pat_rows = inner_rows + 1

    W = (pat_cols + 2 * BORDER_SQUARES) * SQUARE_PX
    H = (pat_rows + 2 * BORDER_SQUARES) * SQUARE_PX

    img = np.full((H, W), 255, dtype=np.uint8)  # white background

    for r in range(pat_rows):
        for c in range(pat_cols):
            if (r + c) % 2 == 1:  # dark squares
                y0 = (r + BORDER_SQUARES) * SQUARE_PX
                x0 = (c + BORDER_SQUARES) * SQUARE_PX
                img[y0:y0 + SQUARE_PX, x0:x0 + SQUARE_PX] = 0

    repo_root = find_repo_root()
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"checkerboard_{inner_cols}x{inner_rows}_bordered.png"

    cv2.imwrite(str(out_path), img)

    print(f"Saved: {out_path}")
    print(f"Image: {W} x {H} px")
    print(f"Pattern: {pat_cols} x {pat_rows} squares, {inner_cols} x {inner_rows} inner corners")
    print()
    print("Next steps:")
    print("  1. Open the PNG on the host (the path above is host-visible via the bind mount)")
    print("  2. Display on a monitor at 100% zoom, full-screen if possible")
    print("  3. Record a 60-90s calibration video varying tilt aggressively")
    print(f"  4. Place the video at {repo_root}/3dgs/capture/calibration_frames/")
    print("  5. Run: python camera_calibration.py")


if __name__ == "__main__":
    main()
