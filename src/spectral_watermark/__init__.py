from .calibration import PilotCalibrator, trim_solid_borders
from .core import SpectralEngine
from .ecc import BitCodec
from .pipeline import WatermarkPipeline

__all__ = [
    "BitCodec",
    "PilotCalibrator",
    "SpectralEngine",
    "WatermarkPipeline",
    "trim_solid_borders",
]
__version__ = "0.1.0"
