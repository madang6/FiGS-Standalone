"""
Base class for vision processors used in flight deployment.

Provides a unified interface for different vision/segmentation models
(CLIPSeg, AM-Radio, etc.) to be used interchangeably in the simulator.

Dependencies are optional - the base class can be imported without torch/cv2,
but actual vision processors require these dependencies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np

# Optional imports for vision processing
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


class VisionProcessorBase(ABC):
    """
    Abstract base class for all vision processors used in flight simulation.

    All vision processors must implement the `process()` method, which takes
    an image and text prompt and returns a tuple of (overlayed_image, scaled_mask).

    Subclasses may also implement `loiter_calibrate()` for hover-based perception tasks.
    """

    def __init__(self, device: Optional[str] = None, cmap: str = "turbo"):
        """
        Initialize the vision processor.

        Args:
            device: Computation device ('cuda' or 'cpu'). Auto-detected if None.
            cmap: Colormap name for visualization
        """
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        except ImportError:
            self.device = "cpu"

        self.cmap = cmap

        # Loiter calibration state
        self.loiter_max = 0.0
        self.loiter_area_frac = 0.0
        self.loiter_mask = None
        self.loiter_contour = None
        self.iou_thresh = 0.50
        self.overlay_alpha = 0.40
        self.overlay_color = (0, 255, 0)   # RGB
        self.loiter_cnt = None
        self.loiter_solidity = None
        self.loiter_eccentricity = None
        self.shape_thresh = 0.20
        self.area_tolerance = 0.15
        self.sol_tol = 0.10
        self.ecc_tol = 0.15

        # Global rescaling state for visualization
        self.running_min = float('inf')
        self.running_max = float('-inf')

    @abstractmethod
    def process(
        self,
        image: Union["Image.Image", np.ndarray],
        prompt: str,
        resize_output_to_input: bool = True,
        use_refinement: bool = False,
        use_smoothing: bool = False,
        scene_change_threshold: float = 1.00,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an image with a text prompt to produce semantic segmentation.

        This is the main inference method that all vision processors must implement.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3) in RGB format
            prompt: Text description of the target object/region to segment
            resize_output_to_input: If True, resize output mask to input image size
            use_refinement: If True, apply superpixel refinement post-processing
            use_smoothing: If True, apply temporal EMA smoothing
            scene_change_threshold: SSIM threshold for scene change detection (1.0 = disabled)
            verbose: If True, print debug information during inference

        Returns:
            Tuple of:
            - overlayed: Colorized segmentation overlaid on original image (H, W, 3) uint8
            - scaled: Normalized segmentation similarity map (H, W) float32 in [0, 1]
        """
        pass

    def loiter_calibrate(
        self,
        logits: np.ndarray,
        frame_img: np.ndarray,
        active_arm: bool = False
    ) -> Tuple[bool, float, float, Optional[np.ndarray]]:
        """
        Calibrate or validate loiter (hover) position based on segmentation output.

        This method tracks a reference region during calibration and validates matches during active hovering.

        Args:
            logits: Segmentation logits/similarity map (H, W)
            frame_img: Original frame image (H, W, 3) RGB
            active_arm: If False, calibration phase; if True, validation phase

        Returns:
            Tuple of:
            - found: Whether a valid match was found (during active_arm phase)
            - sim_score: Maximum similarity score in the frame
            - area_frac: Area fraction of the detected region
            - overlay: Visualization overlay (H, W, 3) RGB or None
        """
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for loiter_calibrate")

        found = False
        H, W = logits.shape
        total_area = H * W
        sim_score = float(logits.max())

        # 1) threshold -> binary mask (per-frame)
        thresh = np.percentile(logits, 90.0)
        mask = (logits >= thresh).astype(np.uint8)  # 0/1

        # 2) connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return found, sim_score, 0.0, None

        # 3) choose the "best" region in this frame:
        best_lab = -1
        best_region_max = -1.0
        best_area = -1
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            region_max = float(logits[labels == lab].max())
            if (region_max > best_region_max) or (region_max == best_region_max and area > best_area):
                best_region_max = region_max
                best_area = area
                best_lab = lab

        # 4) materialize current best region mask (0/255)
        area_frac = float(best_area) / float(total_area)
        curr_region_mask = self._area_targeted_mask(logits, target_frac=area_frac)

        if not active_arm:
            self.overlay_color = (0, 255, 0)
        else:
            self.overlay_color = (0, 0, 255)
        overlay = self._make_overlay(frame_img, curr_region_mask)

        if not active_arm:
            # --- CALIBRATION PHASE ---
            cnt, area_px, solidity, ecc = self._largest_contour_from_mask(curr_region_mask)

            # Should this frame's best become the global reference?
            better = (best_region_max > self.loiter_max) or (
                np.isclose(best_region_max, self.loiter_max, rtol=0, atol=1e-6)
                and (best_area > self.loiter_area_frac * total_area)
            )

            if better and cnt is not None:
                self.loiter_max          = best_region_max
                self.loiter_area_frac    = area_frac
                self.loiter_mask         = curr_region_mask.copy()
                self.loiter_cnt          = cnt
                self.loiter_solidity     = solidity
                self.loiter_eccentricity = ecc

                # Optional visualization of stored outline
                cv2.drawContours(overlay, [self.loiter_cnt], -1, (0, 200, 255), 5)

            return found, sim_score, area_frac, overlay

        # --- ACTIVE / ARM PHASE ---
        if self.loiter_cnt is None:
            return found, sim_score, area_frac, overlay

        # rebuild mask to match reference area
        curr_region_mask = self._area_targeted_mask(logits, target_frac=self.loiter_area_frac)

        # recompute current area fraction from the mask you will compare
        cur_area_frac = np.count_nonzero(curr_region_mask) / float(total_area)

        cur_cnt, cur_area_px, cur_sol, cur_ecc = self._largest_contour_from_mask(curr_region_mask)
        if cur_cnt is None:
            return found, sim_score, cur_area_frac, overlay

        # 1) Area sanity using cur_area_frac
        area_ok = abs(cur_area_frac - self.loiter_area_frac) <= self.area_tolerance * self.loiter_area_frac

        # 2) Shape distance (lower is more similar)
        d = self._match_shape_distance(self.loiter_cnt, cur_cnt)
        shape_ok = (d <= self.shape_thresh)

        # 3) Optional morphology bands
        sol_ok = (abs(cur_sol - self.loiter_solidity) <= self.sol_tol * max(self.loiter_solidity, 1e-6))
        ecc_ok = (abs(cur_ecc - self.loiter_eccentricity) <= self.ecc_tol * max(self.loiter_eccentricity, 1e-6))

        def format_match(val, ref, thr, ok):
            if not ok:
                return "X"
            diff = abs(val - ref)
            close = (1 - diff/thr) * 100 if thr > 0 else 0
            return f"Y{close:2.0f}%" if close > 0 else "Y"

        shp_match = "Y" + f"{(1 - d/self.shape_thresh)*100:2.0f}%" if shape_ok else "X"

        all_matched = all([shape_ok, area_ok, sol_ok, ecc_ok])

        if not active_arm:
            status_prefix = "[LOITER CALIBRATE] "
            if all_matched:
                status_prefix += "O "
            else:
                status_prefix += "o "
        else:
            if all_matched and active_arm:
                status_prefix = "LOITER SUCCESS! "
            else:
                status_prefix = "[LOITER ACTIVE] . "

        print(
            f"{status_prefix}"
            f"shp={d:.3f}/{self.shape_thresh:.3f}/{shp_match} | "
            f"area={cur_area_frac:.3f}/{self.loiter_area_frac:.3f}/"
            f"{format_match(cur_area_frac, self.loiter_area_frac, self.area_tolerance * self.loiter_area_frac, area_ok)} | "
            f"sol={cur_sol:.3f}/{self.loiter_solidity:.3f}/"
            f"{format_match(cur_sol, self.loiter_solidity, self.sol_tol * max(self.loiter_solidity, 1e-6), sol_ok)} | "
            f"ecc={cur_ecc:.3f}/{self.loiter_eccentricity:.3f}/"
            f"{format_match(cur_ecc, self.loiter_eccentricity, self.ecc_tol * max(self.loiter_eccentricity, 1e-6), ecc_ok)}"
        )

        if shape_ok and area_ok and sol_ok and ecc_ok:
            found = True
            rgb_overlay = frame_img.copy()
            fill = np.zeros_like(frame_img); fill[curr_region_mask > 0] = (0, 255, 0)
            rgb_overlay = cv2.addWeighted(fill, 0.4, rgb_overlay, 1.0, 0.0)
            rgb_overlay = cv2.cvtColor(rgb_overlay, cv2.COLOR_BGR2RGB)
            cv2.drawContours(rgb_overlay, [self.loiter_cnt], -1, (0, 165, 255), 2)
            cv2.drawContours(rgb_overlay, [cur_cnt], -1, (0, 255, 0), 2)
            overlay = rgb_overlay

        return found, sim_score, area_frac, overlay

    def _make_overlay(self, frame_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        """Create overlay visualization of mask on frame."""
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for overlay creation")
        overlay = frame_bgr.copy()
        color_img = np.zeros_like(frame_bgr)
        color_img[mask_u8 > 0] = self.overlay_color
        return cv2.addWeighted(color_img, self.overlay_alpha, overlay, 1.0, 0.0)

    def _largest_contour_from_mask(self, mask_u8: np.ndarray):
        """Extract largest contour and compute shape properties."""
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for contour extraction")
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, 0.0, 0.0
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area
        solidity = (area / hull_area) if hull_area > 0 else 1.0

        # Eccentricity via PCA on contour points
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 5:
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
            l1, l2 = float(eigenvalues[0][0]), float(eigenvalues[1][0]) if eigenvalues.shape[0] > 1 else (1.0, 1.0)
            eccentricity = (1.0 - (min(l1,l2) / max(l1,l2))) if max(l1,l2) > 0 else 0.0
        else:
            eccentricity = 0.0
        return cnt, area, solidity, eccentricity

    def _match_shape_distance(self, cnt_ref, cnt_cur) -> float:
        """Compute shape distance between two contours."""
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for shape matching")
        return cv2.matchShapes(cnt_ref, cnt_cur, cv2.CONTOURS_MATCH_I1, 0.0)

    def _area_targeted_mask(self, logits: np.ndarray, target_frac: float,
                            ksize: int = 3, do_open_close: bool = True) -> np.ndarray:
        """
        Build a binary mask whose pixel fraction ~= target_frac by thresholding at the
        corresponding quantile. Then clean it with light morphology.
        Returns 0/255 uint8.
        """
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for mask operations")
        H, W = logits.shape
        total = H * W
        target_frac = float(np.clip(target_frac, 1e-4, 0.90))
        q = 1.0 - target_frac
        t = np.quantile(logits, q)
        mask = (logits >= t).astype(np.uint8) * 255

        if do_open_close:
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        return mask

    def _rescale_global(self, arr: np.ndarray) -> np.ndarray:
        """
        Rescale array to [0,1] using running min/max seen so far.
        Expands the range when new extremes are observed to accommodate all values.
        This prevents uint8 wrapping issues when casting to colormap indices.
        """
        cur_min = float(arr.min())
        cur_max = float(arr.max())

        # Expand range to include all observed values
        self.running_min = min(self.running_min, cur_min)
        self.running_max = max(self.running_max, cur_max)

        span = self.running_max - self.running_min
        if span < 1e-9:
            scaled = np.zeros_like(arr, dtype=np.float32)
        else:
            scaled = (arr - self.running_min) / span
        return scaled


def create_vision_processor(
    processor_type: Optional[str] = None,
    **kwargs
) -> Optional[VisionProcessorBase]:
    """
    Factory function to create vision processor instances.

    Args:
        processor_type: Type of processor to create ('clipseg', 'amradio', or None)
        **kwargs: Additional arguments passed to the processor constructor
                 Common args: device, onnx_model_path, hf_model, radio_version, text_encoder

    Returns:
        VisionProcessor instance or None if processor_type is None or 'none'

    Raises:
        ValueError: If processor_type is not recognized
        ImportError: If required dependencies are not installed
    """
    if processor_type is None or processor_type.lower() == 'none':
        return None

    processor_type_lower = processor_type.lower()

    if processor_type_lower == 'clipseg':
        try:
            from figs.perception.vision_preprocess_clipseg import CLIPSegHFModel
            return CLIPSegHFModel(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"CLIPSeg vision processor requires additional dependencies: {e}\n"
                "Install with: pip install transformers torch pillow opencv-python"
            )

    else:
        raise ValueError(
            f"Unknown vision processor type: '{processor_type}'. "
            f"Supported types: 'clipseg', 'none'"
        )


# ============================================================================
# Shared Utility Functions for All Vision Processors
# ============================================================================

def get_colormap_lut(cmap_name: str = "turbo", lut_size: int = 256) -> np.ndarray:
    """
    Create a lookup table for fast colorization.

    Args:
        cmap_name: Name of matplotlib colormap (default: 'turbo')
        lut_size: Size of the lookup table (default: 256)

    Returns:
        LUT array of shape (lut_size, 3) with RGB colors
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap_name, lut_size)
        lut = (cmap(np.linspace(0, 1, lut_size))[:, :3] * 255).astype(np.uint8)
        return lut
    except ImportError:
        # Fallback: create a simple grayscale LUT
        lut = np.zeros((lut_size, 3), dtype=np.uint8)
        for i in range(lut_size):
            lut[i] = [i, i, i]
        return lut


def colorize_mask_fast(mask_np: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) uint8 mask into a (H, W, 3) RGB image using a lookup table.

    Args:
        mask_np: Input mask array of shape (H, W) with values in [0, lut_size)
        lut: Lookup table of shape (lut_size, 3) from get_colormap_lut()

    Returns:
        Colorized image of shape (H, W, 3) in uint8
    """
    return lut[mask_np]


def blend_overlay_gpu(base: np.ndarray,
                      overlay: np.ndarray,
                      alpha: float = 1.00) -> np.ndarray:
    """
    Blend overlay on base image using GPU acceleration if available.

    Converts `base` to grayscale (if needed), resizes `overlay` on GPU,
    then blends: result = alpha*overlay + (1-alpha)*gray_base.

    Args:
        base: Base image as H x W x 3 BGR/uint8 or H x W grayscale uint8
        overlay: Overlay image as H' x W' x 3 BGR/uint8
        alpha: Opacity of overlay in [0, 1] (default: 1.0)

    Returns:
        Blended result as H x W x 3 BGR/uint8 on CPU

    Raises:
        ValueError: If base or overlay dimensions are incorrect
    """
    try:
        import torch
        import torch.nn.functional as F

        # 1. Upload base to GPU and convert to float
        if base.ndim == 2:
            # Already grayscale
            base_gray = torch.from_numpy(base).float().cuda()
        elif base.ndim == 3 and base.shape[2] == 3:
            # BGR to grayscale using Rec. 601 luma weights
            B = torch.from_numpy(base[:, :, 0]).float().cuda()
            G = torch.from_numpy(base[:, :, 1]).float().cuda()
            R = torch.from_numpy(base[:, :, 2]).float().cuda()
            base_gray = 0.114 * B + 0.587 * G + 0.299 * R
        else:
            raise ValueError("`base` must be H x W (gray) or H x W x 3 (BGR).")

        H, W = base_gray.shape

        # 2. Upload overlay to GPU as float tensor
        if overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError("`overlay` must be H' x W' x 3 (BGR/uint8).")
        ov = torch.from_numpy(overlay).float().permute(2, 0, 1).cuda()
        ov = ov.unsqueeze(0)

        # 3. Resize overlay to match base dimensions
        ov_resized = F.interpolate(
            ov, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)

        # 4. Stack grayscale to 3 channels
        gray3 = base_gray.unsqueeze(0).repeat(3, 1, 1)

        # 5. Blend on GPU
        blended = alpha * ov_resized + (1.0 - alpha) * gray3

        # 6. Clamp to [0,255], cast to uint8, move to CPU
        blended = blended.clamp(0, 255).round().byte()
        return blended.permute(1, 2, 0).cpu().numpy()

    except ImportError:
        # Fallback to CPU implementation
        if not HAS_CV2:
            raise ImportError("Either torch or opencv-python is required for blending")

        if base.ndim == 2:
            base_gray = base
        elif base.ndim == 3 and base.shape[2] == 3:
            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("`base` must be H x W (gray) or H x W x 3 (BGR).")

        H, W = base_gray.shape

        if overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError("`overlay` must be H' x W' x 3 (BGR/uint8).")

        # Resize overlay to match base
        overlay_resized = cv2.resize(overlay, (W, H))

        # Stack grayscale to 3 channels
        gray3 = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

        # Blend
        blended = cv2.addWeighted(overlay_resized, alpha, gray3, 1.0 - alpha, 0)
        return blended


def fast_superpixel_seeds(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply SEEDS superpixel refinement to a segmentation mask.

    Uses superpixel segmentation to refine mask by taking the majority label
    within each superpixel region.

    Args:
        image: Input RGB image of shape (H, W, 3) in uint8
        mask: Input mask of shape (H, W) in uint8

    Returns:
        Refined mask of same shape as input
    """
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for superpixel refinement")

    h, w = image.shape[:2]
    num_superpixels = 400
    num_levels = 4
    prior = 2
    histogram_bins = 5
    double_step = False

    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        w, h, image.shape[2],
        num_superpixels, num_levels, prior,
        histogram_bins, double_step
    )

    seeds.iterate(image, num_iterations=2)
    labels = seeds.getLabels()
    out = np.zeros_like(mask)

    for label in np.unique(labels):
        region = mask[labels == label]
        if region.size > 0:
            out[labels == label] = np.bincount(region).argmax()

    return out
