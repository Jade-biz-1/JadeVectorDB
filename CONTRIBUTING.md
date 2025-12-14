# Contributing to JadeVectorDB

Thank you for your interest in contributing to JadeVectorDB! We welcome contributions from the community to help improve our high-performance distributed vector database.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Coding Standards](#coding-standards)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Getting Started

To get started with contributing to JadeVectorDB:

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/JadeVectorDB.git
   ```
3. Follow the setup instructions in our [Developer Guide](BOOTSTRAP.md) to set up your development environment.

## How to Contribute

### Reporting Bugs

Before submitting a bug report, please check if the issue has already been reported. If not, create a new issue following our template, providing:

- A clear and descriptive title
- A detailed description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- System information (OS, compiler version, etc.)

### Suggesting Enhancements

Feature requests are welcome! Before submitting an enhancement suggestion:

1. Check if a similar idea already exists in our roadmap or issues
2. If not, create a new issue with:
   - A clear and descriptive title
   - A detailed explanation of the proposed feature
   - Benefits and use cases for the feature
   - Any potential drawbacks or considerations

### Submitting Pull Requests

1. **Create a Branch**: Create a branch for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**: Implement your feature or fix following our [Development Guidelines](#development-guidelines).

3. **Write Tests**: Ensure your changes are covered by appropriate tests.

4. **Update Documentation**: If your changes affect functionality, update the relevant documentation.

5. **Follow Coding Standards**: Ensure your code follows our coding standards.

6. **Commit Your Changes**: Write clear, descriptive commit messages following the conventional commit format.

7. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request**: Submit a pull request to the main repository with:
   - A clear title and description
   - References to any related issues
   - Screenshots or examples if applicable

## Development Guidelines

### Coding Standards

JadeVectorDB is primarily written in C++20 with performance as a key consideration. Please follow these guidelines:

- Follow modern C++ best practices (C++20 features where appropriate)
- Use meaningful variable and function names
- Write clear comments for complex logic
- Follow the existing code style in the project
- Ensure thread safety where applicable
- Optimize for performance while maintaining code readability

### Testing

- Write unit tests for new functionality using Google Test
- Ensure all tests pass before submitting a pull request
- Include integration tests for major features
- Maintain or improve code coverage

To run tests:
```bash
cd backend/build
./jadevectordb_tests
```

### Documentation

- Update or add documentation for any API changes
- Document public interfaces and major components
- Keep README and other documentation up-to-date with your changes

## Community

If you have questions or need help, feel free to:

1. Open an issue on GitHub
2. Reach out to the maintainers

Thank you for helping make JadeVectorDB better!