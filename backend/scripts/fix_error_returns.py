#!/usr/bin/env python3
import re

files = [
    "src/api/grpc/distributed_master_client.cpp",
    "src/api/grpc/distributed_worker_service.cpp"
]

for filename in files:
    with open(filename, 'r') as f:
        content = f.read()

    # Fix: return tl::make_unexpected(ErrorHandler::create_error(...);
    # Should be: return tl::make_unexpected(ErrorHandler::create_error(...));

    # Match lines with return tl::make_unexpected(ErrorHandler::create_error
    # and add closing paren before semicolon
    content = re.sub(
        r'return tl::make_unexpected\(ErrorHandler::create_error\(([^;]+)\);',
        r'return tl::make_unexpected(ErrorHandler::create_error(\1));',
        content
    )

    with open(filename, 'w') as f:
        f.write(content)

    print(f"Fixed {filename}")
