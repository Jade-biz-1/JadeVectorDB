#!/usr/bin/env python3
import re
import sys

def fix_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Fix patterns:
    # 1. Remove extra )); at end of tl::unexpected(*.error()) lines
    content = re.sub(r'return tl::unexpected\(([^)]+\.error\(\))\)\);', r'return tl::unexpected(\1);', content)

    # 2. Fix ErrorHandler::create_error lines missing closing paren
    content = re.sub(r'tl::make_unexpected\(ErrorHandler::create_error\(([^;]+)\);', r'tl::make_unexpected(ErrorHandler::create_error(\1));', content)

    # 3. Fix logger lines with extra parens
    content = re.sub(r'(logger_->[a-z]+\([^)]+\))\);', r'\1);', content)

    # 4. Fix futures.push_back and results.push_back
    content = re.sub(r'(futures\.push_back\(std::move\(future\)\))\);', r'\1;', content)
    content = re.sub(r'(results\.push_back\(future\.get\(\)\))\);', r'\1;', content)

    # 5. Fix std::min with extra parens
    content = re.sub(r'(std::min\([^)]+\))\);', r'\1;', content)

    with open(filename, 'w') as f:
        f.write(content)
    print(f"Fixed {filename}")

if __name__ == "__main__":
    files = [
        "src/services/distributed_query_planner.cpp",
        "src/services/distributed_query_executor.cpp",
        "src/services/distributed_write_coordinator.cpp"
    ]

    for file in files:
        try:
            fix_file(file)
        except Exception as e:
            print(f"Error fixing {file}: {e}")
