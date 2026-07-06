# Contributing guidelines

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
    * [Reporting Bugs](#reporting-bugs)
    * [Suggesting Enhancements](#suggesting-enhancements)
    * [Pull Requests](#pull-requests)
3. [Development Environment Setup](#development-environment-setup)
4. [Coding Style & Guidelines](#coding-style--guidelines)
5. [Getting Help](#getting-help)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and welcoming environment for everyone. Please ensure that all communications in issues, pull requests, and discussions remain professional and collaborative.

## How Can I Contribute?

### Reporting Bugs

If you find a bug in the source code, you can help us by submitting an issue to our GitHub Repository. Even better, you can submit a Pull Request with a fix.

When filing an issue, please include:
* A clear and descriptive title.
* The version of CARE and Python you are using.
* Your operating system (e.g., Ubuntu).
* A minimal, reproducible example demonstrating the problem.
* The exact error message or traceback, if applicable.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing workflow, please open an issue first to discuss it with the maintainers. This ensures your idea aligns with the project's scope before you spend time writing code.

### Pull Requests

The process for submitting a Pull Request (PR) is as follows:

1. **Fork** the repo and create your branch from `main`.
2. **Implement** your feature or bug fix.
3. **Test** your changes thoroughly. Ensure that existing tests pass and write new tests for any new functionality.
4. **Document** your code. If you've added a new feature, update the relevant documentation and include Python docstrings.
5. **Ensure the test suite passes** before opening the PR.
6. **Issue the PR!** Provide a clear description of the problem you are solving and the solution you have implemented. 

## Development Environment Setup

To set up your local development environment:

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/care.git
   cd care
   ```
3. Create a virtual environment (using `venv` or `conda`):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the package in editable mode along with development dependencies and optional ML evaluation dependencies:
   ```bash
   pip install -e .[dev,fairchemv2]  # example for installing with FairChemV2 ML evaluator
   ```

## Coding Style & Guidelines

* **Language:** Python 3.12.
* **Style:** Please adhere to PEP 8 guidelines. We recommend using tools like `black` and `flake8` for formatting and linting.
* **Typing:** Use Python type hints where possible to make the codebase more readable and maintainable.
* **Docstrings:** Use consistent docstrings (e.g., Sphinx/NumPy style) for all new modules, classes, and functions. 

## Getting Help

If you have any questions regarding the codebase, the contribution process, or microkinetic modeling workflows within the framework, feel free to reach out via GitHub Issues.

Thank you for your contributions!
