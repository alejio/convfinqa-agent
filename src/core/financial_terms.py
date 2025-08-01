"""
Shared financial terms and constants for consistent recognition across the codebase.

This module consolidates financial terminology that was previously duplicated across
multiple modules (conversation.py, agent.py, table_analyzer.py, query_parser.py, tools.py).
"""


class FinancialTerms:
    """Centralized financial terminology for consistent recognition."""

    # Core financial metrics and line items
    CORE_METRICS = [
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
        "cash flow",
        "margin",
        "dividend",
        "interest",
        "tax",
        "investment",
    ]

    # Financial statement types
    STATEMENT_TYPES = [
        "balance_sheet",
        "income_statement",
        "cash_flow_statement",
        "financial_table",
    ]

    # Time period identifiers
    TIME_PERIODS = [
        "quarter",
        "year",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]

    # Analysis and calculation terms
    ANALYSIS_TERMS = [
        "ratio",
        "change",
        "growth",
        "variance",
        "percentage",
        "total",
        "net",
        "gross",
        "operating",
        "fiscal",
    ]

    @classmethod
    def get_all_terms(cls) -> set[str]:
        """Get all financial terms as a set for efficient lookup (lowercase)."""
        all_terms = (
            cls.CORE_METRICS
            + cls.STATEMENT_TYPES
            + cls.TIME_PERIODS
            + cls.ANALYSIS_TERMS
        )
        return set(term.lower() for term in all_terms)

    @classmethod
    def get_core_terms(cls) -> set[str]:
        """Get core financial metrics only (lowercase)."""
        return set(term.lower() for term in cls.CORE_METRICS)

    @classmethod
    def get_analysis_terms(cls) -> set[str]:
        """Get analysis and calculation terms (lowercase)."""
        return set(term.lower() for term in cls.ANALYSIS_TERMS)

    @classmethod
    def is_financial_term(cls, term: str) -> bool:
        """Check if a term is a known financial term (case-insensitive)."""
        return term.lower() in cls.get_all_terms()

    @classmethod
    def extract_financial_terms(cls, text: str) -> set[str]:
        """Extract all financial terms found in text (case-insensitive)."""
        text_lower = text.lower()
        found_terms = set()

        # Create a mapping from lowercase to original case
        original_terms = (
            cls.CORE_METRICS
            + cls.STATEMENT_TYPES
            + cls.TIME_PERIODS
            + cls.ANALYSIS_TERMS
        )
        term_mapping = {term.lower(): term for term in original_terms}

        for term_lower in cls.get_all_terms():
            if term_lower in text_lower:
                # Return the original case version
                found_terms.add(term_mapping[term_lower])

        return found_terms

    @classmethod
    def has_financial_content(cls, text: str) -> bool:
        """Check if text contains any financial terms."""
        text_lower = text.lower()
        return any(term in text_lower for term in cls.get_all_terms())
