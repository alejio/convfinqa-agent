"""
Shared numeric extraction and conversion utilities.

This module consolidates numeric processing logic that was previously duplicated across
multiple modules (metrics.py, dspy_extraction.py, tools.py).
"""

import re


class NumericExtractor:
    """Centralized numeric value extraction and processing."""

    @staticmethod
    def extract_primary_numeric_value(text: str) -> str | None:
        """
        Extract the primary numeric value from text using comprehensive patterns.

        This consolidates the complex regex patterns from ExecutionAccuracyMetric._extract_numeric_value()
        and provides consistent extraction behavior across the codebase.

        Args:
            text: Input text to extract numeric value from

        Returns:
            Extracted numeric value as string, or None if no value found
        """
        if not text:
            return None

        # Clean the text
        text = text.strip()

        # Strategy 1: Look for final_answer() tool output or clean numeric line
        lines = text.split("\n")
        for line in reversed(lines):  # Check from end backwards
            line = line.strip()
            if not line:
                continue
            # Look for patterns like "60.94" or "-4" on their own line
            if re.match(r"^[-+]?\d*\.?\d+$", line):
                return line
            # Look for final_answer output
            if "final_answer" in line.lower() and ":" in line:
                parts = line.split(":")[-1].strip()
                if re.match(r"^[-+]?\d*\.?\d+$", parts):
                    return parts

        # Strategy 2: Enhanced regex for comprehensive numeric extraction
        # This pattern handles: $123.45, -123, (123), 123.45%, 1,234.56
        # Order matters - more specific patterns first
        patterns = [
            r"(?:^|\s)(\([-+]?\d+(?:,\d{3})*(?:\.\d+)?\))",  # (123.45) - parentheses for negatives
            r"\$?([-+]?\d{1,3}(?:,\d{3})*\.\d+)%?",  # Decimals with dollars/% like $123.45 or 60.94
            r"\$?([-+]?\d*\.\d+)",  # Decimals like .45 or 123.45
            r"\$?([-+]?\d{1,3}(?:,\d{3})*)%?",  # Large integers with commas
            r"([-+]?\d+)",  # Regular integers (but avoid years)
        ]

        best_match = ""
        best_length = 0

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = str(match).replace(",", "").replace("$", "").replace("%", "")

                # Handle parentheses negatives: (123) -> -123
                if cleaned.startswith("(") and cleaned.endswith(")"):
                    cleaned = "-" + cleaned[1:-1]

                # Skip if it's not a valid number
                try:
                    num_value = float(cleaned)

                    # Skip likely years (1900-2100) unless they're the only option
                    if 1900 <= abs(num_value) <= 2100 and "." not in cleaned:
                        # Only use years if nothing else found
                        if not best_match:
                            continue

                    # Prefer decimal numbers over integers when both are found
                    current_score = len(cleaned.replace(".", "").replace("-", ""))
                    if "." in cleaned:
                        current_score += 10  # Boost score for decimals

                    if current_score > best_length:
                        best_match = cleaned
                        best_length = current_score
                except ValueError:
                    continue

        return best_match if best_match else None

    @staticmethod
    def extract_basic_numeric_value(text: str) -> str | None:
        """
        Basic numeric extraction fallback when comprehensive extraction isn't needed.

        Consolidates the simpler extraction logic from _extract_numeric_value_basic().
        """
        if not text:
            return None

        # Clean the text
        text = text.strip()

        # Remove common prefixes and suffixes
        text = text.replace("$", "").replace(",", "").strip()

        # Try to extract a number using basic patterns
        # Look for standalone numbers (including negative and decimals)
        number_pattern = r"-?\d+\.?\d*"
        numbers = re.findall(number_pattern, text)

        if numbers:
            # Return the last number found (often the final answer)
            return str(numbers[-1])

        return None

    @staticmethod
    def safe_float_conversion(
        value: str | float, default: float | None = None
    ) -> float | None:
        """
        Safely convert a value to float with consistent error handling.

        Consolidates the repeated try-catch float conversion patterns found throughout the codebase.

        Args:
            value: Value to convert (string or already float)
            default: Default value to return on conversion failure

        Returns:
            Converted float value or default
        """
        if value is None:
            return default

        if isinstance(value, float):
            return value

        try:
            # Handle string values
            if isinstance(value, str):
                # Clean common formatting
                cleaned = (
                    value.strip().replace(",", "").replace("$", "").replace("%", "")
                )

                # Handle parentheses negatives: (123) -> -123
                if cleaned.startswith("(") and cleaned.endswith(")"):
                    cleaned = "-" + cleaned[1:-1]

                return float(cleaned)
            else:
                return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def clean_numeric_string(value: str) -> str:
        """
        Clean a numeric string by removing common formatting characters.

        Args:
            value: String to clean

        Returns:
            Cleaned string with formatting removed
        """
        if not isinstance(value, str):
            return str(value)

        # Remove common formatting
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")

        # Handle parentheses negatives: (123) -> -123
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        return cleaned

    @staticmethod
    def numeric_match(actual_num: str | None, expected_num: str | None) -> bool:
        """
        Check if two numeric values match exactly with floating point tolerance.

        Consolidates the numeric matching logic from ExecutionAccuracyMetric._numeric_match().
        """
        # If we can't extract numbers from either, fall back to string comparison
        if actual_num is None or expected_num is None:
            return actual_num is None and expected_num is None

        try:
            # Convert to floats and compare with small tolerance for floating point errors
            actual_float = float(actual_num)
            expected_float = float(expected_num)
            return abs(actual_float - expected_float) < 1e-6
        except ValueError:
            # If conversion fails, fall back to string comparison
            return actual_num == expected_num

    @staticmethod
    def convert_to_float_with_formatting(value: str | float | int) -> float:
        """
        Convert value to float handling all financial formatting patterns.

        Consolidates all the numeric conversion logic scattered across tools.py,
        calculate_change, and other modules.

        Args:
            value: Value to convert (can be string, float, or int)

        Returns:
            Converted float value

        Raises:
            ValueError: If value cannot be converted to float
        """
        if isinstance(value, int | float):
            return float(value)

        if not isinstance(value, str):
            value = str(value)

        # Remove whitespace
        value = value.strip()

        if not value:
            raise ValueError("Empty string cannot be converted to float")

        # Handle common financial formatting
        # Remove currency symbols, commas, percentage signs
        cleaned = value.replace("$", "").replace(",", "").replace("%", "")

        # Handle parentheses notation for negative numbers: (123) -> -123
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        # Try conversion
        try:
            return float(cleaned)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{value}' to float: {e}") from e

    @staticmethod
    def clean_for_display(value: str | float | int) -> str:
        """
        Clean numeric value for display purposes.

        Consolidates the cleaning logic from final_answer tool and other display functions.

        Args:
            value: Value to clean

        Returns:
            Cleaned string representation
        """
        if isinstance(value, int | float):
            return str(value)

        if not isinstance(value, str):
            value = str(value)

        # Remove common prefixes/suffixes while preserving the number
        cleaned = value.strip()
        cleaned = cleaned.replace("$", "").replace("%", "").replace(",", "")

        # Validate it's a number
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            # If we can't parse it as a number, return original stripped value
            return value.strip()
