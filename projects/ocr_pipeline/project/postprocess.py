"""Text postprocessing for OCR output."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PostprocessConfig:
    """Postprocessing configuration."""

    normalize_whitespace: bool = True
    strip_text: bool = True
    lowercase: bool = False


class TextPostprocessor:
    """Postprocessor for OCR text output."""

    def __init__(self, config: PostprocessConfig | None = None) -> None:
        """Initialize postprocessor.

        Args:
            config: Postprocessing configuration.
        """
        self.config = config or PostprocessConfig()

    def __call__(self, text: str) -> str:
        """Apply postprocessing.

        Args:
            text: Raw OCR output text.

        Returns:
            Cleaned text.
        """
        return self.process(text)

    def process(self, text: str) -> str:
        """Apply full postprocessing pipeline.

        Args:
            text: Raw OCR output text.

        Returns:
            Cleaned text.
        """
        if self.config.strip_text:
            text = text.strip()

        if self.config.normalize_whitespace:
            text = self.normalize_whitespace(text)

        if self.config.lowercase:
            text = text.lower()

        return text

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# Field-specific extractors for SROIE
class SROIEFieldExtractor:
    """Extract structured fields from SROIE receipt text."""

    # Common patterns for SROIE fields
    DATE_PATTERNS = [
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",  # DD-MM-YYYY, DD/MM/YYYY
        r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}",  # YYYY-MM-DD
        r"\d{1,2}\s+\w+\s+\d{2,4}",  # DD Month YYYY
    ]

    TOTAL_PATTERNS = [
        r"(?:total|amount|sum)[:\s]*\$?\s*([\d,]+\.?\d*)",
        r"\$\s*([\d,]+\.\d{2})",
        r"([\d,]+\.\d{2})\s*$",
    ]

    @classmethod
    def extract_date(cls, text: str) -> str | None:
        """Extract date from text.

        Args:
            text: Input text.

        Returns:
            Extracted date string or None.
        """
        for pattern in cls.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    @classmethod
    def extract_total(cls, text: str) -> str | None:
        """Extract total amount from text.

        Args:
            text: Input text.

        Returns:
            Extracted total string or None.
        """
        for pattern in cls.TOTAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the captured group if present, else full match
                return match.group(1) if match.groups() else match.group(0)
        return None

    @classmethod
    def normalize_date(cls, date_str: str) -> str:
        """Normalize date to standard format.

        Args:
            date_str: Raw date string.

        Returns:
            Normalized date string (YYYY-MM-DD).
        """
        # Simple normalization - production code would use dateutil
        # Replace common separators
        normalized = date_str.replace("/", "-")

        # Try to parse and reformat
        # This is a placeholder - real implementation would be more robust
        return normalized

    @classmethod
    def normalize_total(cls, total_str: str) -> str:
        """Normalize total amount to standard format.

        Args:
            total_str: Raw total string.

        Returns:
            Normalized total (numeric string with 2 decimal places).
        """
        # Remove currency symbols and commas
        cleaned = re.sub(r"[$,]", "", total_str)

        # Ensure 2 decimal places
        try:
            value = float(cleaned)
            return f"{value:.2f}"
        except ValueError:
            return cleaned


def compute_field_confidence(
    prediction: str,
    field_type: str,
) -> float:
    """Compute confidence score for a field prediction.

    Uses heuristics based on field type to estimate confidence.

    Args:
        prediction: Predicted text.
        field_type: Type of field (date, total, company, address).

    Returns:
        Confidence score between 0 and 1.
    """
    confidence = 0.5  # Base confidence

    if field_type == "date":
        # Higher confidence if matches date pattern
        for pattern in SROIEFieldExtractor.DATE_PATTERNS:
            if re.search(pattern, prediction, re.IGNORECASE):
                confidence = 0.8
                break

    elif field_type == "total":
        # Higher confidence if matches currency pattern
        if re.search(r"\d+\.\d{2}", prediction):
            confidence = 0.8

    elif field_type == "company":
        # Higher confidence for longer names
        if len(prediction) > 5:
            confidence = 0.7

    elif field_type == "address":
        # Higher confidence if contains address keywords
        address_keywords = ["street", "road", "avenue", "building", "floor", "unit"]
        if any(kw in prediction.lower() for kw in address_keywords):
            confidence = 0.7

    return confidence
