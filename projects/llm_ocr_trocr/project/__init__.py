"""TrOCR OCR Pipeline project module."""

from projects.llm_ocr_trocr.project.data import SROIEDataset
from projects.llm_ocr_trocr.project.model import TrOCRWrapper

__all__ = ["SROIEDataset", "TrOCRWrapper"]
