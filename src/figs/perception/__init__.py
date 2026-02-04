"""
Perception module for FiGS-Standalone.

This module provides vision processing capabilities for semantic segmentation
and object detection during flight simulation. The module is designed to work
with optional dependencies - core functionality works without vision dependencies,
and richer features become available when torch/transformers are installed.
"""

from figs.perception.vision_processor_base import (
    VisionProcessorBase,
    create_vision_processor,
)

__all__ = [
    'VisionProcessorBase',
    'create_vision_processor',
]
