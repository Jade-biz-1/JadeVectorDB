#!/usr/bin/env python3
import re

files = [
    "src/services/distributed_query_planner.cpp",
    "src/services/distributed_query_executor.cpp",
    "src/services/distributed_write_coordinator.cpp"
]

for filename in files:
    with open(filename, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        # Fix tl::make_unexpected with ErrorHandler::create_error - single line, remove triple parens at end
        if 'tl::make_unexpected(ErrorHandler::create_error' in line and line.rstrip().endswith(')));'):
            # Check if it's a complete single-line statement (has both opening and closing)
            if line.count('(') == line.count(')'):
                # This is a complete statement with extra parens, remove one set
                line = line.rstrip()[:-1] + '\n'  # Remove one )

        # Fix tl::unexpected(*.error()) - should have only one closing paren
        if 'tl::unexpected(' in line and '.error())' in line and line.rstrip().endswith(');'):
            # Already correct
            pass
        elif 'tl::unexpected(' in line and '.error())' in line and line.rstrip().endswith('));'):
            # Has extra paren
            line = line.rstrip()[:-1] + '\n'

        fixed_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(fixed_lines)

    print(f"Fixed {filename}")
