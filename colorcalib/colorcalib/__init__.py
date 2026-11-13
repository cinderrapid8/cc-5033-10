"""Cross-camera color calibration for product photography."""

from .charts import Chart, builtin_colorchecker24, load_chart_json
from .detection import DetectionResult, DetectionError, detect_chart, extract_manual
from .pipeline import (
    CalibrationResult,
    InsufficientSamplesError,
    IllConditionedError,
    calibrate,
    apply_mapping_to_rgb,
    load_mapping,
)
from .apply import apply_batch

__all__ = [
    "Chart",
    "builtin_colorchecker24",
    "load_chart_json",
    "DetectionResult",
    "DetectionError",
    "detect_chart",
    "extract_manual",
    "CalibrationResult",
    "InsufficientSamplesError",
    "IllConditionedError",
    "calibrate",
    "apply_mapping_to_rgb",
    "load_mapping",
    "apply_batch",
]
