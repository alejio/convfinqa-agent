"""
DSPy-powered financial term extraction replacing static lists.

This module consolidates financial terminology extraction using intelligent DSPy signatures
instead of hardcoded term lists across multiple modules.
"""

import dspy


class FinancialTermExtraction(dspy.Signature):
    """Extract and classify financial terms from text using intelligent understanding."""

    text: str = dspy.InputField(desc="Text to analyze for financial terms")
    context: str = dspy.InputField(
        desc="Additional context about the domain or purpose"
    )

    financial_terms: str = dspy.OutputField(
        desc="Comma-separated list of financial terms found"
    )
    term_categories: str = dspy.OutputField(
        desc="Categories: revenue, expense, asset, liability, equity, performance, time"
    )
    confidence: str = dspy.OutputField(desc="Confidence in extraction: high/medium/low")


class EntityExtraction(dspy.Signature):
    """Extract relevant entities from financial conversation text."""

    text: str = dspy.InputField(desc="Text from financial conversation or analysis")
    conversation_context: str = dspy.InputField(
        desc="Context from previous conversation turns"
    )

    entities: str = dspy.OutputField(
        desc="Comma-separated list of important entities referenced"
    )
    entity_types: str = dspy.OutputField(
        desc="Types: financial_metric, time_period, table_reference, calculation"
    )
    relationships: str = dspy.OutputField(desc="How entities relate to each other")


class FinancialTerms:
    """DSPy-powered financial terminology extraction and classification."""

    def __init__(self) -> None:
        """Initialize DSPy extractors."""
        self.term_extractor = dspy.ChainOfThought(FinancialTermExtraction)
        self.entity_extractor = dspy.ChainOfThought(EntityExtraction)

        # Import existing DSPy classifiers from query_parser
        from ..functions.query_parser import (
            FinancialMetricClassifier,
            TimeColumnClassifier,
        )

        self.metric_classifier = dspy.Predict(FinancialMetricClassifier)
        self.time_classifier = dspy.Predict(TimeColumnClassifier)

    def extract_financial_terms(self, text: str, context: str = "") -> set[str]:
        """Extract financial terms using DSPy intelligence."""
        try:
            result = self.term_extractor(text=text, context=context)
            if result.financial_terms:
                terms = [term.strip() for term in result.financial_terms.split(",")]
                return set(terms)
        except Exception:
            # Fallback to basic term recognition
            pass

        return self._fallback_extract_terms(text)

    def extract_entities(self, text: str, conversation_context: str = "") -> list[str]:
        """Extract entities using DSPy understanding."""
        try:
            result = self.entity_extractor(
                text=text, conversation_context=conversation_context
            )
            if result.entities:
                entities = [entity.strip() for entity in result.entities.split(",")]
                return entities
        except Exception:
            # Fallback to basic extraction
            pass

        return self._fallback_extract_entities(text)

    def is_financial_term(self, term: str, context: str = "") -> bool:
        """Check if term is financial using DSPy classification."""
        try:
            result = self.metric_classifier(label=term, context=context)
            return str(result.is_financial_metric).lower() == "true"
        except Exception:
            return self._fallback_is_financial(term)

    def is_time_column(self, column_name: str, context: str = "") -> bool:
        """Check if column represents time using DSPy classification."""
        try:
            result = self.time_classifier(column_name=column_name, context=context)
            return str(result.is_time_column).lower() == "true"
        except Exception:
            return self._fallback_is_time(column_name)

    def has_financial_content(self, text: str) -> bool:
        """Check if text contains financial content."""
        extracted_terms = self.extract_financial_terms(text)
        return len(extracted_terms) > 0

    # Fallback methods for when DSPy fails
    def _fallback_extract_terms(self, text: str) -> set[str]:
        """Basic fallback term extraction."""
        text_lower = text.lower()
        basic_terms = [
            "revenue",
            "sales",
            "income",
            "profit",
            "loss",
            "expenses",
            "costs",
            "assets",
            "liabilities",
            "equity",
            "debt",
            "cash",
            "margin",
        ]
        found = set()
        for term in basic_terms:
            if term in text_lower:
                found.add(term)
        return found

    def _fallback_extract_entities(self, text: str) -> list[str]:
        """Basic fallback entity extraction."""
        entities = []
        text_lower = text.lower()
        if "table" in text_lower:
            entities.append("financial_table")
        return entities

    def _fallback_is_financial(self, term: str) -> bool:
        """Basic fallback financial term check."""
        term_lower = term.lower()
        basic_terms = {
            "revenue",
            "sales",
            "income",
            "profit",
            "loss",
            "expenses",
            "costs",
        }
        return any(basic in term_lower for basic in basic_terms)

    def _fallback_is_time(self, column_name: str) -> bool:
        """Basic fallback time column check."""
        name_lower = column_name.lower()
        return any(
            indicator in name_lower
            for indicator in ["year", "quarter", "q1", "q2", "q3", "q4"]
        )


# Global instance for backward compatibility
_financial_terms_instance = None


def get_financial_terms_instance() -> FinancialTerms:
    """Get global FinancialTerms instance."""
    global _financial_terms_instance
    if _financial_terms_instance is None:
        _financial_terms_instance = FinancialTerms()
    return _financial_terms_instance


# Compatibility methods for existing code
def extract_financial_terms(text: str) -> set[str]:
    """Compatibility wrapper for extract_financial_terms."""
    return get_financial_terms_instance().extract_financial_terms(text)


def is_financial_term(term: str) -> bool:
    """Compatibility wrapper for is_financial_term."""
    return get_financial_terms_instance().is_financial_term(term)


def has_financial_content(text: str) -> bool:
    """Compatibility wrapper for has_financial_content."""
    return get_financial_terms_instance().has_financial_content(text)
