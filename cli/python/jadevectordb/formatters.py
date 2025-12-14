"""
Output formatters for JadeVectorDB CLI

This module provides formatting functions for different output formats
(JSON, YAML, table) to improve CLI usability and integration.
"""

import json
import sys
from typing import Any, Dict, List, Optional

def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as JSON

    :param data: Data to format
    :param indent: Number of spaces for indentation
    :return: JSON formatted string
    """
    return json.dumps(data, indent=indent)

def format_yaml(data: Any) -> str:
    """
    Format data as YAML

    :param data: Data to format
    :return: YAML formatted string
    """
    try:
        import yaml
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install PyYAML", file=sys.stderr)
        print("Falling back to JSON output:", file=sys.stderr)
        return format_json(data)

def format_table(data: Any, headers: Optional[List[str]] = None) -> str:
    """
    Format data as a table

    :param data: Data to format (dict, list of dicts, or list of lists)
    :param headers: Optional custom headers for table columns
    :return: Table formatted string
    """
    try:
        from tabulate import tabulate

        # Handle different data types
        if isinstance(data, dict):
            # Single dictionary - show as key-value pairs
            table_data = [[k, v] for k, v in data.items()]
            return tabulate(table_data, headers=headers or ["Key", "Value"], tablefmt="grid")

        elif isinstance(data, list):
            if not data:
                return "No data to display"

            # List of dictionaries - show as table
            if isinstance(data[0], dict):
                if headers is None:
                    headers = list(data[0].keys())
                table_data = [[item.get(h, '') for h in headers] for item in data]
                return tabulate(table_data, headers=headers, tablefmt="grid")

            # List of lists - show as table
            else:
                return tabulate(data, headers=headers or [], tablefmt="grid")

        else:
            # For other types, convert to string
            return str(data)

    except ImportError:
        print("Warning: tabulate not installed. Install with: pip install tabulate", file=sys.stderr)
        print("Falling back to JSON output:", file=sys.stderr)
        return format_json(data)

def format_output(data: Any, output_format: str = 'json', headers: Optional[List[str]] = None) -> str:
    """
    Format data according to specified output format

    :param data: Data to format
    :param output_format: Output format (json, yaml, table)
    :param headers: Optional headers for table format
    :return: Formatted string
    """
    output_format = output_format.lower()

    if output_format == 'json':
        return format_json(data)
    elif output_format == 'yaml':
        return format_yaml(data)
    elif output_format == 'table':
        return format_table(data, headers)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def print_formatted(data: Any, output_format: str = 'json', headers: Optional[List[str]] = None):
    """
    Print data in specified format to stdout

    :param data: Data to format and print
    :param output_format: Output format (json, yaml, table)
    :param headers: Optional headers for table format
    """
    formatted = format_output(data, output_format, headers)
    print(formatted)
