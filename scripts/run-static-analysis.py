#!/usr/bin/env python3

"""
Script to run static analysis on the JadeVectorDB C++ codebase.
This script runs clang-tidy and cppcheck to identify potential issues
and ensure C++20 compliance across all modules.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False  # We'll handle return codes ourselves
        )
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        else:
            print("Command completed successfully")
            
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)


def run_clang_tidy(source_dir):
    """Run clang-tidy on the source code."""
    print("\n" + "="*60)
    print("RUNNING CLANG-TIDY ANALYSIS")
    print("="*60)
    
    # Check if clang-tidy is available
    result = subprocess.run(["which", "clang-tidy"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: clang-tidy not found. Please install clang-tidy first.")
        return False
    
    # Find all .cpp and .h files in the source directory
    source_files = []
    for ext in ["*.cpp", "*.h", "*.hpp", "*.cc", "*.cxx"]:
        source_files.extend(Path(source_dir).rglob(ext))
    
    print(f"Found {len(source_files)} source files to analyze")
    
    # Create a compilation database if it doesn't exist
    build_dir = Path(source_dir) / "build"
    if not build_dir.exists():
        print("Build directory doesn't exist, creating it and generating compilation database...")
        build_dir.mkdir(exist_ok=True)
        cmake_result = subprocess.run(
            ["cmake", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ".."], 
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if cmake_result.returncode != 0:
            print(f"Error generating compilation database: {cmake_result.stderr}")
            return False
    
    # Run clang-tidy on each source file
    issues_found = 0
    for src_file in source_files:
        print(f"Analyzing {src_file}")
        cmd = [
            "clang-tidy",
            str(src_file),
            f"-p={build_dir}",  # Use compilation database
            "--warnings-as-errors=*",  # Treat warnings as errors for compliance
        ]
        
        return_code, stdout, stderr = run_command(cmd, f"Clang-tidy on {src_file}", source_dir)
        
        if return_code != 0:
            issues_found += 1
            print(f"Found issues in {src_file}")
            print("Output:", stdout)
            if stderr:
                print("Errors:", stderr)
    
    print(f"\nClang-tidy analysis completed. Issues found in {issues_found} files.")
    return True


def run_cppcheck(source_dir):
    """Run cppcheck on the source code."""
    print("\n" + "="*60)
    print("RUNNING CPPCHECK ANALYSIS")
    print("="*60)
    
    # Check if cppcheck is available
    result = subprocess.run(["which", "cppcheck"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: cppcheck not found. Please install cppcheck first.")
        return False
    
    # Define the cppcheck command
    cmd = [
        "cppcheck",
        "--enable=warning,style,performance,portability,information",
        "--std=c++20",
        "--verbose",
        "--check-library",
        f"--project={source_dir}/build/compile_commands.json" if (Path(source_dir) / "build" / "compile_commands.json").exists() else source_dir + "/src",
        "--xml",
        "--xml-version=2"
    ]
    
    # Also output to a file
    output_file = Path(source_dir) / "cppcheck_report.xml"
    cmd.extend(["--output-file=" + str(output_file)])
    
    return_code, stdout, stderr = run_command(cmd, "Cppcheck analysis", source_dir)
    
    if return_code == 0:
        print("Cppcheck completed successfully")
    else:
        print("Cppcheck found issues (this is expected)")
        
    # Also run without XML for console output
    cmd_console = [
        "cppcheck",
        "--enable=warning,style,performance,portability,information",
        "--std=c++20",
        "--verbose",
        "--check-library",
        f"--project={source_dir}/build/compile_commands.json" if (Path(source_dir) / "build" / "compile_commands.json").exists() else source_dir + "/src"
    ]
    
    return_code, stdout, stderr = run_command(cmd_console, "Cppcheck analysis (console)", source_dir)
    
    print(f"Cppcheck analysis completed.")
    if stdout:
        print("Output:", stdout)
    if stderr:
        print("Errors:", stderr)
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Run static analysis on JadeVectorDB codebase")
    parser.add_argument(
        "--source-dir", 
        type=str, 
        default="backend", 
        help="Path to source directory (default: backend)"
    )
    parser.add_argument(
        "--tool",
        type=str,
        choices=["clang-tidy", "cppcheck", "both"],
        default="both",
        help="Which tool to run (default: both)"
    )
    
    args = parser.parse_args()
    
    # Verify source directory exists
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"Error: Source directory {args.source_dir} does not exist")
        return 1
    
    success = True
    
    if args.tool in ["clang-tidy", "both"]:
        success = run_clang_tidy(args.source_dir) and success
    
    if args.tool in ["cppcheck", "both"]:
        success = run_cppcheck(args.source_dir) and success
    
    print("\n" + "="*60)
    print("STATIC ANALYSIS COMPLETED")
    print("="*60)
    
    if success:
        print("Static analysis scripts executed successfully.")
        print("Please review the output and generated reports to address any issues found.")
        return 0
    else:
        print("Static analysis encountered errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())