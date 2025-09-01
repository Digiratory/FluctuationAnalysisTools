# Contribution Guidelines

Thank you for your interest in contributing to StatTools! This document provides guidelines to help you get started and ensure your contributions are effective and aligned with the project's standards.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- A GitHub account

### Setting Up the Development Environment

1. **Fork the Repository**: If you don't have write access, fork the repository by clicking the "Fork" button on the GitHub page.

2. **Clone the Repository**: Clone your fork (or the main repository if you have access) to your local machine:
   ```bash
   git clone https://github.com/your-username/StatTools.git
   cd StatTools
   ```

3. **Create a Branch**: Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install Dependencies**: Install the package in editable mode along with development dependencies:
   ```bash
   pip install -e .
   ```

5. **Set Up Pre-Commit Hooks**: Install pre-commit hooks to ensure code quality:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Making Changes

1. **Write Code**: Make your changes following the coding standards below.
2. **Run Pre-Commit**: Before committing, run pre-commit to check your code:
   ```bash
   pre-commit run --all-files
   ```
3. **Test Your Changes**: Run the test suite to ensure everything works:
   ```bash
   python -m pytest tests/
   ```
4. **Commit Your Changes**: Use clear, descriptive commit messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
5. **Push and Create Pull Request**: Push your branch and create a pull request on GitHub.

### Coding Standards

- **Code Formatting**: Use Black for code formatting and isort for import sorting. These are enforced by pre-commit hooks.
- **Style Guide**: Follow PEP 8 conventions.
- **Documentation**: Add docstrings to new functions and classes. Update documentation as needed.
- **Type Hints**: Use type hints where appropriate.

### Commit Messages

- Use the present tense (e.g., "Add feature" not "Added feature").
- Keep the first line under 50 characters.
- Be descriptive but concise.
- Reference issues if applicable (e.g., "Fix issue #123").

### Testing

- Write tests for new features in the `tests/` directory.
- Ensure all tests pass before submitting a pull request.
- Run tests locally: `python -m pytest tests/`

## Pull Requests

- Provide a clear description of the changes.
- Reference any related issues.
- Ensure CI checks pass.
- Request review from maintainers.

## Issues and Bugs

- Use GitHub Issues to report bugs or suggest features.
- Provide detailed information: steps to reproduce, expected vs. actual behavior, environment details.
- Check existing issues before creating new ones.

## Additional Resources

- [README.md](README.md) for project overview and usage examples.
- [CHANGELOG.md](CHANGELOG.md) for version history.

By following these guidelines, you help maintain the quality and consistency of the StatTools project. Happy contributing!
