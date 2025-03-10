import pytest

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that interact with Parquet files."
    )

def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless --run-integration is specified."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")