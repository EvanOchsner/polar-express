#!/usr/bin/env python3
"""
Test runner script for the JSONPath to Polars converter.
"""
import sys
import argparse
import subprocess


def main():
    """Run tests for the JSONPath to Polars converter."""
    parser = argparse.ArgumentParser(description="Run tests for JSONPath to Polars converter")
    parser.add_argument("--integration", action="store_true", default=False, 
                        help="Run integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Run tests in verbose mode")
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Run a specific test (e.g., 'test_simple_field_access')")
    args = parser.parse_args()
    
    # Build the command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.integration:
        cmd.append("--run-integration")
    
    if args.test:
        if "/" in args.test or "::" in args.test:
            # Assume the user provided a path or path::function
            cmd.append(args.test)
        else:
            # Try to find the test in either test file
            cmd.append(f"tests/test_jsonpath_conversion.py::TestJsonPathToPolars::{args.test}")
            cmd.append(f"tests/test_integration.py::TestJsonPathIntegration::{args.test}")
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())