"""
Tests for data loading functionality
"""


import pytest
from rich import print as rich_print

from src.core.logger import get_logger
from src.data.loader import create_data_loader

logger = get_logger(__name__)


@pytest.fixture
def data_loader():
    """Create a data loader instance for testing"""
    return create_data_loader()


@pytest.fixture
def dataset(data_loader):
    """Load dataset for testing"""
    return data_loader.load_dataset()


@pytest.fixture
def first_record(data_loader, dataset):
    """Get the first record from the dataset"""
    first_record_id = dataset.train[0].id
    return data_loader.get_record(first_record_id)


@pytest.mark.unittest
def test_data_loader_creation():
    """Test that data loader can be created successfully"""
    data_loader = create_data_loader()
    assert data_loader is not None
    rich_print("✓ Data loader created successfully")


@pytest.mark.unittest
def test_dataset_loading(data_loader):
    """Test that dataset can be loaded successfully"""
    dataset = data_loader.load_dataset()

    assert len(dataset.train) > 0
    assert len(dataset.dev) > 0

    rich_print(
        f"✓ Dataset loaded: {len(dataset.train)} train + {len(dataset.dev)} dev records"
    )


@pytest.mark.unittest
def test_record_retrieval(data_loader, dataset):
    """Test that records can be retrieved by ID"""
    first_record_id = dataset.train[0].id
    record = data_loader.get_record(first_record_id)

    assert record is not None
    assert record.id == first_record_id

    rich_print(f"✓ Record retrieval works: {record.id}")


@pytest.mark.unittest
def test_table_extraction(first_record):
    """Test that financial tables can be extracted from records"""
    table = first_record.get_financial_table()
    df = table.to_dataframe()

    assert df.shape[0] > 0
    assert df.shape[1] > 0

    rich_print(f"✓ Table extraction works: {df.shape[0]} rows × {df.shape[1]} columns")


@pytest.mark.unittest
def test_schema_detection(first_record):
    """Test that table schema can be detected"""
    table = first_record.get_financial_table()
    table_schema = table.table_schema

    assert len(table_schema.columns) > 0

    rich_print(
        f"✓ Schema detection works: {len(table_schema.columns)} columns detected"
    )


@pytest.mark.integration
def test_milestone_1_integration(data_loader, dataset, first_record):
    """Integration test for all Milestone 1 functionality"""
    rich_print("[bold]Testing Milestone 1: Data Ingestion & Schema[/bold]")

    # Test data loader creation
    assert data_loader is not None
    rich_print("✓ Data loader created successfully")

    # Test dataset loading
    assert len(dataset.train) > 0
    assert len(dataset.dev) > 0
    rich_print(
        f"✓ Dataset loaded: {len(dataset.train)} train + {len(dataset.dev)} dev records"
    )

    # Test record retrieval
    assert first_record is not None
    rich_print(f"✓ Record retrieval works: {first_record.id}")

    # Test table extraction
    table = first_record.get_financial_table()
    df = table.to_dataframe()
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    rich_print(f"✓ Table extraction works: {df.shape[0]} rows × {df.shape[1]} columns")

    # Test schema detection
    table_schema = table.table_schema
    assert len(table_schema.columns) > 0
    rich_print(
        f"✓ Schema detection works: {len(table_schema.columns)} columns detected"
    )

    rich_print("\n[bold green]✓ Milestone 1 implementation successful![/bold green]")
