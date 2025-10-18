# Security Testing Framework

This document explains how to set up and run security testing for the JadeVectorDB project.

## Setting up Security Testing Tools

First, install the required security testing tools:

```bash
./scripts/setup-security-tools.sh
```

This script will install various security testing tools including:
- nmap: Network exploration and security auditing
- nikto: Web server scanner
- sqlmap: SQL injection detection
- gosec: Go security scanner
- snyk: Dependency vulnerability scanner

## Running Security Tests

To run a comprehensive security assessment:

```bash
python3 scripts/run-security-tests.py --project-dir backend
```

To run security tests with specific targets:

```bash
# Scan a specific web application endpoint
python3 scripts/run-security-tests.py --target-url http://localhost:8080 --skip-web-scan

# Scan a specific host and port range
python3 scripts/run-security-tests.py --target-host 192.168.1.1 --port-range 1-65535
```

## Components of the Security Testing Framework

### 1. Source Code Security Analysis
- Checks for hardcoded credentials
- Identifies potential SQL injection vulnerabilities
- Detects command injection patterns
- Finds path traversal vulnerabilities

### 2. Dependency Vulnerability Scanning
- Uses Snyk to scan for known vulnerabilities in dependencies
- Checks for outdated or insecure packages
- Provides remediation advice

### 3. Network Security Scanning
- Uses nmap to identify open ports and services
- Detects potentially exposed services
- Checks for common security misconfigurations

### 4. Web Application Security Scanning
- Uses Nikto to scan for web vulnerabilities
- Identifies common web security issues
- Checks for exposed files and directories

### 5. Go Code Security Scanning
- Uses gosec to analyze Go code for security issues
- Checks for common Go security anti-patterns

## Integration with Security Hardening Task

This security testing framework is part of the broader Security Hardening (T182) task, which aims to:
- Implement comprehensive security features beyond basic authentication
- Add advanced security mechanisms
- Perform security testing and validation

## Security Testing Workflow

1. Set up tools using the setup script
2. Run comprehensive security assessment
3. Review reports and identified vulnerabilities
4. Address high-priority security issues
5. Re-run tests to verify fixes
6. Document security measures implemented

## Continuous Security Testing

Security tests should be integrated into the CI/CD pipeline to:
- Catch security issues early
- Ensure new code doesn't introduce vulnerabilities
- Maintain security posture during development

## Security Reporting

The security testing framework generates reports in multiple formats:
- JSON summary: `security-assessment-summary.json`
- Snyk report: Integrated in the output
- Nikto report: `nikto-report.txt` (if web scan is run)
- Gosec report: `gosec-report.json` (if Go code is present)

## Advanced Security Testing

For more comprehensive security validation, consider:
- Penetration testing after initial automated scans
- Manual code review of security-critical components
- Security audit of authentication and authorization mechanisms
- Data protection and privacy compliance verification