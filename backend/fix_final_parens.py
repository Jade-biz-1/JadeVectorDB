#!/usr/bin/env python3
import re

files = [
    "src/services/distributed_query_planner.cpp",
    "src/services/distributed_query_executor.cpp",
    "src/services/distributed_write_coordinator.cpp"
]

for filename in files:
    with open(filename, 'r') as f:
        content = f.read()

    # Fix all ErrorHandler::create_error calls that end with )));
    # They should end with ));
    content = re.sub(
        r'(tl::make_unexpected\(ErrorHandler::create_error\([^;]+)\)\)\);',
        r'\1));',
        content
    )

    # Fix logger calls with extra parens
    content = re.sub(
        r'(logger_->(?:trace|debug|info|warn|error|fatal)\([^)]+\))\)\);',
        r'\1);',
        content
    )

    with open(filename, 'w') as f:
        f.write(content)

    print(f"Fixed {filename}")
