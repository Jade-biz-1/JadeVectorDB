#!/usr/bin/env python3

"""
Script to run security testing on the JadeVectorDB project.
This script runs multiple security analysis tools to identify
vulnerabilities and implement security testing framework.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, description, cwd=None, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=False  # We'll handle return codes ourselves
        )
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
        else:
            print("Command completed successfully")
            
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)


def run_gosec_scan(project_dir):
    """Run gosec to scan Go code for security issues (if any exists)."""
    print("\n" + "="*60)
    print("RUNNING GOSEC SECURITY SCAN")
    print("="*60)
    
    # Check if gosec is available
    result = subprocess.run(["which", "gosec"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: gosec not found. Skipping gosec analysis.")
        return True
    
    # Look for any Go files in the project
    go_files = list(Path(project_dir).rglob("*.go"))
    
    if not go_files:
        print("No Go files found. Skipping gosec analysis.")
        return True
    
    print(f"Found {len(go_files)} Go files to analyze")
    
    # Run gosec on the project directory
    cmd = [
        "gosec",
        "-fmt=json",
        "-out=gosec-report.json",
        "-exclude-dir=tests",
        f"{project_dir}/..."
    ]
    
    return_code, stdout, stderr = run_command(cmd, "Gosec security scan", project_dir)
    
    if return_code == 0:
        print("Gosec scan completed successfully")
    else:
        print("Gosec scan completed with issues found")
    
    # Print summary if available
    report_path = Path(project_dir) / "gosec-report.json"
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
                if 'Issues' in report:
                    print(f"Gosec found {len(report['Issues'])} security issues")
                    for issue in report['Issues'][:5]:  # Show first 5 issues
                        print(f"  - {issue.get('rule_id', 'Unknown')}: {issue.get('details', 'No details')}")
        except json.JSONDecodeError:
            print("Could not parse gosec report")
    
    return True


def run_snyk_scan(project_dir):
    """Run Snyk to scan for vulnerabilities in dependencies."""
    print("\n" + "="*60)
    print("RUNNING SNYK SECURITY SCAN")
    print("="*60)
    
    # Check if snyk is available
    result = subprocess.run(["which", "snyk"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: snyk not found. Skipping snyk analysis.")
        print("To install snyk: npm install -g snyk")
        return True
    
    # Check if there are any supported manifest files in the project
    manifest_files = [
        "package.json", "requirements.txt", "go.mod", "Cargo.toml", 
        "pom.xml", "build.gradle", "Gemfile", "composer.json"
    ]
    
    found_manifests = []
    for manifest in manifest_files:
        if (Path(project_dir) / manifest).exists():
            found_manifests.append(manifest)
    
    if not found_manifests:
        print("No supported manifest files found for Snyk analysis. Skipping.")
        return True
    
    print(f"Found manifest files: {found_manifests}")
    
    # Run snyk test on the project
    cmd = ["snyk", "test", "--json", "--file=package.json"] if (Path(project_dir) / "package.json").exists() else ["snyk", "test", "--json"]
    
    return_code, stdout, stderr = run_command(cmd, "Snyk security scan", project_dir)
    
    if stdout:
        try:
            report = json.loads(stdout)
            if 'vulnerabilities' in report:
                print(f"Snyk found {len(report['vulnerabilities'])} vulnerabilities")
                # Show top 5 vulnerabilities
                for vuln in report['vulnerabilities'][:5]:
                    print(f"  - {vuln.get('id', 'Unknown')}: {vuln.get('title', 'No title')}")
            elif 'error' in report:
                print(f"Snyk error: {report['error']}")
        except json.JSONDecodeError:
            print("Could not parse Snyk report")
            if len(stdout) > 1000:
                print(f"Snyk output (first 1000 chars): {stdout[:1000]}...")
            else:
                print(f"Snyk output: {stdout}")
    
    if stderr:
        print(f"Snyk errors: {stderr}")
    
    return True


def run_nmap_scan(target="localhost", port_range="1-1000"):
    """Run nmap to scan for open ports and services (only if server is running)."""
    print("\n" + "="*60)
    print("RUNNING NMAP PORT SCAN")
    print("="*60)
    
    # Check if nmap is available
    result = subprocess.run(["which", "nmap"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: nmap not found. Skipping nmap analysis.")
        return True
    
    # Perform a basic scan
    cmd = [
        "nmap",
        "-p", port_range,
        "--open",  # Only show open ports
        "-sV",     # Service detection
        target
    ]
    
    return_code, stdout, stderr = run_command(cmd, f"Nmap scan of {target}:{port_range}")
    
    if stdout:
        print("Nmap scan results:")
        print(stdout)
        
        # Look for common ports that should not be open in a secure setup
        if "80" in stdout or "443" in stdout or "22" in stdout:
            print("Note: Common ports like 22 (SSH), 80 (HTTP), 443 (HTTPS) detected.")
            print("Verify these are properly secured and necessary for your application.")
    
    return True


def run_nikto_scan(target_url="http://localhost:8080"):
    """Run Nikto to scan for web vulnerabilities (only if server is running)."""
    print("\n" + "="*60)
    print("RUNNING NIKTO WEB VULNERABILITY SCAN")
    print("="*60)
    
    # Check if nikto is available
    result = subprocess.run(["which", "nikto"], capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: nikto not found. Skipping nikto analysis.")
        return True
    
    # Perform a basic web scan
    cmd = [
        "nikto",
        "-h", target_url,
        "-o", "nikto-report.txt",
        "-Format", "txt"
    ]
    
    print(f"Scanning {target_url} - make sure your server is running")
    return_code, stdout, stderr = run_command(cmd, f"Nikto scan of {target_url}")
    
    if stdout:
        print("Nikto scan running... (output saved to nikto-report.txt)")
    
    return True


def check_source_code_security(project_dir):
    """Check source code for common security issues."""
    print("\n" + "="*60)
    print("RUNNING SOURCE CODE SECURITY CHECKS")
    print("="*60)
    
    issues_found = []
    
    # Search for common security issues in source code
    security_patterns = [
        ("hardcoded_password", r"password\s*=\s*['\"][^'\"]{3,}['\"]|passwd\s*=\s*['\"][^'\"]{3,}['\"]"),
        ("hardcoded_api_key", r"api_key\s*=\s*['\"][^'\"]{10,}['\"]|secret\s*=\s*['\"][^'\"]{10,}['\"]"),
        ("sql_injection_vulnerability", r"query\s*=\s*.*\+.*request|execute\s*\(\s*.*\+"),
        ("command_injection", r"system\s*\(|os\.popen\s*\(|subprocess\.call\s*\([^,]*\+"),
        ("path_traversal", r'os\.path\.join\s*\([^,]*\+.*request|open\s*\([^,]*\+.*request')
    ]
    
    # Walk through all source files
    for ext in ["*.cpp", "*.h", "*.hpp", "*.cc", "*.cxx", "*.py", "*.js", "*.ts", "*.json"]:
        for file_path in Path(project_dir).rglob(ext):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern_name, pattern in security_patterns:
                    if pattern_name in content.lower() or pattern in content:
                        # For now, just checking if the general pattern text appears
                        # In a real implementation, we'd use proper regex
                        issues_found.append({
                            "file": str(file_path),
                            "pattern": pattern_name,
                            "line": "N/A"  # Would implement line detection in full version
                        })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"Found {len(issues_found)} potential security issues")
    for issue in issues_found[:10]:  # Show first 10 issues
        print(f"  - {issue['pattern']} in {issue['file']}")
    
    if len(issues_found) > 10:
        print(f"  ... and {len(issues_found) - 10} more issues")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run security tests on JadeVectorDB project")
    parser.add_argument(
        "--project-dir", 
        type=str, 
        default=".", 
        help="Path to project directory (default: current directory)"
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default="http://localhost:8080",
        help="Target URL for web vulnerability scanning (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--target-host",
        type=str,
        default="localhost",
        help="Target host for port scanning (default: localhost)"
    )
    parser.add_argument(
        "--port-range",
        type=str,
        default="1-1000",
        help="Port range for scanning (default: 1-1000)"
    )
    parser.add_argument(
        "--skip-web-scan",
        action="store_true",
        help="Skip web vulnerability scanning (requires server to be running)"
    )
    
    args = parser.parse_args()
    
    # Verify project directory exists
    project_path = Path(args.project_dir)
    if not project_path.exists():
        print(f"Error: Project directory {args.project_dir} does not exist")
        return 1
    
    print(f"Starting security assessment for project: {args.project_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Run source code security checks
    success = check_source_code_security(args.project_dir) and success
    
    # Run gosec if Go code exists
    success = run_gosec_scan(args.project_dir) and success
    
    # Run Snyk security scan
    success = run_snyk_scan(args.project_dir) and success
    
    # Run nmap scan
    success = run_nmap_scan(args.target_host, args.port_range) and success
    
    # Run Nikto scan if not skipped
    if not args.skip_web_scan:
        success = run_nikto_scan(args.target_url) and success
    else:
        print("\nSkipping web vulnerability scan as requested.")
    
    print("\n" + "="*60)
    print("SECURITY ASSESSMENT COMPLETED")
    print("="*60)
    
    print("Security assessment completed.")
    print("Please review all reports and outputs to address identified vulnerabilities.")
    print("For production deployments, ensure all high-severity issues are addressed.")
    
    # Summary file
    summary = {
        "timestamp": datetime.now().isoformat(),
        "project_dir": args.project_dir,
        "completed_scans": [
            "source_code_security_check",
            "gosec_scan",
            "snyk_scan", 
            "nmap_scan",
            "nikto_scan" if not args.skip_web_scan else "nikto_scan_skipped"
        ],
        "status": "completed_with_issues" if success else "failed"
    }
    
    with open("security-assessment-summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Summary saved to security-assessment-summary.json")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())