"""
CLIPSeg vision processor implementation for FiGS-Standalone.

This module provides the CLIPSegHFModel class for semantic segmentation
using the CIDAS/clipseg model from Hugging Face.

Requirements:
    - torch
    - transformers
    - PIL/Pillow
    - opencv-python
    - (optional) onnxruntime for ONNX inference
"""

import os
import time
import numpy as np

from typing import Optional, Tuple, Union

# Check for required dependencies
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None

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

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    CLIPSegProcessor = None
    CLIPSegForImageSegmentation = None

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ort = None

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    ssim = None

from figs.perception.vision_processor_base import (
    VisionProcessorBase,
    get_colormap_lut,
    colorize_mask_fast,
    blend_overlay_gpu,
    fast_superpixel_seeds
)


def _check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    if not HAS_TORCH:
        missing.append("torch")
    if not HAS_PIL:
        missing.append("Pillow")
    if not HAS_CV2:
        missing.append("opencv-python")
    if not HAS_TRANSFORMERS:
        missing.append("transformers")

    if missing:
        raise ImportError(
            f"CLIPSegHFModel requires the following packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


class CLIPSegHFModel(VisionProcessorBase):
    """
    HuggingFace CLIPSeg wrapper for torch inference, with optional ONNX fallback.
    """

    def __init__(
        self,
        hf_model: str = "CIDAS/clipseg-rd64-refined",
        device: Optional[str] = None,
        cmap: str = "turbo",
        onnx_model_path: Optional[str] = None,
        onnx_model_fp16_path: Optional[str] = None
    ):
        """
        Initialize CLIPSeg model.

        Args:
            hf_model: HuggingFace model identifier
            device: Computation device ('cuda' or 'cpu')
            cmap: Colormap for visualization
            onnx_model_path: Path to ONNX model (optional, for faster inference)
            onnx_model_fp16_path: Path to FP16 ONNX model (optional)
        """
        _check_dependencies()

        # Initialize base class
        super().__init__(device=device, cmap=cmap)

        # Load HF processor & model
        self.processor = CLIPSegProcessor.from_pretrained(hf_model, use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_model)
        self.model.to(self.device).eval()

        # Color LUT for visualization
        self.lut = get_colormap_lut(cmap_name=cmap)

        # Caches and state
        self.prev_image = None
        self.prev_output = None
        self.segmentation_ema = None
        self.ema_alpha = 0.7
        self.last_superpixel_mask = None
        self.superpixel_every = 1
        self.frame_counter = 0

        # ONNX support
        self.use_onnx = False
        self.ort_session = None
        self.using_fp16 = False

        if onnx_model_path is not None:
            if not HAS_ONNX:
                raise ImportError(
                    "onnxruntime is not installed; pip install onnxruntime to use ONNX inference."
                )
            # If the .onnx file is missing, export it
            if not os.path.isfile(onnx_model_path):
                self._export_onnx(onnx_model_path)

            if onnx_model_fp16_path:
                if not os.path.isfile(onnx_model_fp16_path):
                    self._convert_to_fp16(onnx_model_path, onnx_model_fp16_path)
                onnx_model_path = onnx_model_fp16_path
                self.using_fp16 = True

            # Load the ONNX session
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.log_severity_level = 1

            self.ort_session = ort.InferenceSession(
                onnx_model_path,
                sess_options=so,
                providers=["CUDAExecutionProvider"]
            )
            self.io_binding = self.ort_session.io_binding()
            self.use_onnx = True

            self._io_binding = None
            self._input_gpu = None
            self._output_gpu = None

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
        Process an image with CLIPSeg for semantic segmentation.

        This is the unified interface method that implements VisionProcessorBase.
        """
        return self.clipseg_hf_inference(
            image=image,
            prompt=prompt,
            resize_output_to_input=resize_output_to_input,
            use_refinement=use_refinement,
            use_smoothing=use_smoothing,
            scene_change_threshold=scene_change_threshold,
            verbose=verbose
        )

    def _export_onnx(self, onnx_path: str):
        """Export the HF CLIPSeg model to ONNX."""
        dummy_prompt = "a photo of a cat"
        dummy_img = Image.new("RGB", (224, 224), color="white")

        torch_inputs = self.processor(
            images=dummy_img,
            text=dummy_prompt,
            return_tensors="pt"
        )
        torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

        torch.onnx.export(
            self.model,
            (
                torch_inputs["input_ids"],
                torch_inputs["pixel_values"],
                torch_inputs["attention_mask"],
            ),
            onnx_path,
            input_names=["input_ids", "pixel_values", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {1: "seq_len"},
                "attention_mask": {1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    def _convert_to_fp16(self, onnx_path: str, fp16_path: str):
        """Convert an ONNX model to FP16."""
        import onnx
        from onnxconverter_common import float16

        model = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
        )
        onnx.save(model_fp16, fp16_path)

    def _run_onnx_model(self, img: "Image.Image", prompt: str) -> np.ndarray:
        """Run inference using the ONNX model."""
        # Preprocess on GPU
        torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
        if self.using_fp16:
            torch_inputs = {
                k: (v.half().to(self.device) if k == "pixel_values" else v.to(self.device))
                for k, v in torch_inputs.items()
            }
        else:
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}

        # Fresh IOBinding
        io_binding = self.ort_session.io_binding()

        # Bind inputs
        sess_input_names = {inp.name for inp in self.ort_session.get_inputs()}
        for name, tensor in torch_inputs.items():
            if name not in sess_input_names:
                continue
            elem_type = np.float32 if tensor.dtype == torch.float32 else np.int64
            io_binding.bind_input(
                name=name,
                device_type=self.device,
                device_id=0,
                element_type=elem_type,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        # Figure out the ONNX output shape
        out_meta = self.ort_session.get_outputs()[0]
        B, _, H, W = torch_inputs["pixel_values"].shape
        if len(out_meta.shape) == 3:
            out_shape = (B, H, W)
        elif len(out_meta.shape) == 4:
            C = out_meta.shape[1] if isinstance(out_meta.shape[1], int) else 1
            out_shape = (B, C, H, W)
        else:
            raise RuntimeError(f"Unsupported logits rank: {len(out_meta.shape)}")

        # Allocate & bind output buffer on GPU
        out_dtype = torch.float16 if self.using_fp16 else torch.float32
        output_gpu = torch.empty(out_shape, dtype=out_dtype, device=self.device)
        io_binding.bind_output(
            name=out_meta.name,
            device_type=self.device,
            device_id=0,
            element_type=(np.float16 if self.using_fp16 else np.float32),
            shape=out_shape,
            buffer_ptr=output_gpu.data_ptr(),
        )

        # Run
        self.ort_session.run_with_iobinding(io_binding)

        # Fetch & squeeze
        result = output_gpu.cpu().numpy()
        if result.ndim == 4 and result.shape[1] == 1:
            result = result[:, 0]
        return result.squeeze()

    def clipseg_hf_inference(
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
        Run CLIPSeg on a PIL image or numpy array.

        Returns:
            Tuple of (overlayed_image, scaled_mask)
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # --- Step 1: Normalize input to PIL + NumPy ---
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            image_np = image
        elif isinstance(image, Image.Image):
            img = image
            image_np = np.array(image)
        else:
            raise TypeError(f"Unsupported image type {type(image)}")

        # --- Step 2: Determine whether to reuse ---
        should_reuse = False
        if scene_change_threshold < 1.0 and self.prev_image is not None and self.prev_output is not None:
            if HAS_SKIMAGE and HAS_CV2:
                prev_small = cv2.resize(self.prev_image, (64, 64))
                curr_small = cv2.resize(image_np, (64, 64))
                prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                ssim_score = ssim(prev_gray, curr_gray, data_range=1.0)
                should_reuse = ssim_score >= scene_change_threshold
                log(f"[DEBUG] SSIM = {ssim_score:.4f}, Threshold = {scene_change_threshold}, Reuse = {should_reuse}")

        # --- Step 3: Reuse path (warp previous mask) ---
        if should_reuse:
            mask_u8 = self._warp_mask(self.prev_image, image_np, self.prev_output)
            mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)
            colorized = colorize_mask_fast(mask_u8, self.lut)
            overlayed = blend_overlay_gpu(image_np, colorized)
            return overlayed, mask_u8.astype(np.float32) / 255.0

        # --- Step 4: Run inference ---
        start = time.time()
        if self.use_onnx:
            arr = self._run_onnx_model(img, prompt)
        else:
            torch_inputs = self.processor(images=img, text=prompt, return_tensors="pt")
            torch_inputs = {k: v.to(self.device) for k, v in torch_inputs.items()}
            with torch.no_grad():
                logits = self.model(**torch_inputs).logits
            arr = logits.cpu().squeeze().numpy().astype(np.float32)

        # Apply sigmoid and scale
        prob = 1.0 / (1.0 + np.exp(-arr))

        # Track running max for consistent visualization
        cur_max = float(arr.max())
        self._max_logit = max(getattr(self, "_max_logit", cur_max), cur_max)
        global_max_prob = 1.0 / (1.0 + np.exp(-self._max_logit))
        scaled = prob / (global_max_prob + 1e-8)
        scaled = np.clip(scaled, 0.0, 1.0)
        mask_u8 = (scaled * 255).astype(np.uint8)

        # Also create logits-based mask for visualization
        logits_scaled = self._rescale_global(arr)
        logits_mask_u8 = (logits_scaled * 255).astype(np.uint8)

        if resize_output_to_input:
            mask_u8 = np.array(Image.fromarray(mask_u8).resize(img.size, resample=Image.BILINEAR))
            scaled = np.array(Image.fromarray((scaled * 255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR)).astype(np.float32) / 255.0

        # --- Step 5: Post-processing ---
        if use_smoothing:
            mask_f = mask_u8.astype(np.float32)
            if self.segmentation_ema is None or self.segmentation_ema.shape != mask_f.shape:
                self.segmentation_ema = mask_f.copy()
            else:
                self.segmentation_ema = (
                    self.ema_alpha * mask_f + (1 - self.ema_alpha) * self.segmentation_ema
                )
            mask_u8 = np.clip(self.segmentation_ema, 0, 255).astype(np.uint8)

            if HAS_CV2:
                mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=75, sigmaSpace=75)

        if use_refinement and HAS_CV2:
            self.frame_counter += 1
            if self.frame_counter % self.superpixel_every == 0:
                try:
                    self.last_superpixel_mask = fast_superpixel_seeds(image_np, mask_u8)
                except Exception:
                    pass  # Skip if ximgproc not available
            if self.last_superpixel_mask is not None:
                mask_u8 = self.last_superpixel_mask

        # --- Step 6: Render and cache ---
        colorized = colorize_mask_fast(logits_mask_u8, self.lut)
        overlayed = blend_overlay_gpu(image_np, colorized)

        self.prev_image = image_np.copy()
        self.prev_output = mask_u8.copy()

        end = time.time()
        log(f"CLIPSeg inference time: {end - start:.3f} seconds")
        return overlayed, scaled

    def _warp_mask(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
        """Warp previous mask to current frame using optical flow."""
        if not HAS_CV2:
            return prev_mask

        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        h, w = prev_mask.shape
        flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
        remap = flow_map + flow
        warped = cv2.remap(
            prev_mask, remap[..., 0], remap[..., 1],
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        return warped.astype(np.uint8)
