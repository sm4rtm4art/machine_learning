"""OCR Pipeline project module."""

from projects.ocr_pipeline.project.data import SROIEDataset
from projects.ocr_pipeline.project.model import TrOCRWrapper

__all__ = ["SROIEDataset", "TrOCRWrapper"]
