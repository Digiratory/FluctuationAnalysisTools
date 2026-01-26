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

Contributors MUST follow the Conventional Commits 1.0.0 specification for all commit messages. The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this section are to be interpreted as described in RFC 2119.

#### Format
Commits MUST be prefixed with a type, which consists of a noun, feat, fix, etc., followed by the OPTIONAL scope, OPTIONAL !, and REQUIRED terminal colon and space.

#### Types
- The type `feat` MUST be used when a commit adds a new feature to your application or library.
- The type `fix` MUST be used when a commit represents a bug fix for your application.
- Types other than `feat` and `fix` MAY be used in your commit messages, e.g., `docs: update ref docs`.

#### Scope
A scope MAY be provided after a type. A scope MUST consist of a noun describing a section of the codebase surrounded by parenthesis, e.g., `fix(parser):`.

#### Description
A description MUST immediately follow the colon and space after the type/scope prefix. The description is a short summary of the code changes, e.g., `fix: array parsing issue when multiple spaces were contained in string`.

#### Body
A longer commit body MAY be provided after the short description, providing additional contextual information about the code changes. The body MUST begin one blank line after the description. A commit body is free-form and MAY consist of any number of newline separated paragraphs.

#### Footer
One or more footers MAY be provided one blank line after the body. Each footer MUST consist of a word token, followed by either a `:<space>` or `<space>#` separator, followed by a string value (this is inspired by the git trailer convention).

A footer's token MUST use `-` in place of whitespace characters, e.g., `Acked-by` (this helps differentiate the footer section from a multi-paragraph body). An exception is made for `BREAKING CHANGE`, which MAY also be used as a token.

A footer's value MAY contain spaces and newlines, and parsing MUST terminate when the next valid footer token/separator pair is observed.

#### Breaking Changes
Breaking changes MUST be indicated in the type/scope prefix of a commit, or as an entry in the footer.

If included as a footer, a breaking change MUST consist of the uppercase text `BREAKING CHANGE`, followed by a colon, space, and description, e.g., `BREAKING CHANGE: environment variables now take precedence over config files`.

If included in the type/scope prefix, breaking changes MUST be indicated by a `!` immediately before the `:`. If `!` is used, `BREAKING CHANGE:` MAY be omitted from the footer section, and the commit description SHALL be used to describe the breaking change.

#### Case Sensitivity
The units of information that make up Conventional Commits MUST NOT be treated as case sensitive by implementors, with the exception of `BREAKING CHANGE` which MUST be uppercase.

`BREAKING-CHANGE` MUST be synonymous with `BREAKING CHANGE`, when used as a token in a footer.

#### Examples
- `feat: add new generator for fractional Brownian motion`
- `fix(analysis): resolve memory leak in DFA calculation`
- `docs: update API documentation for filters module`
- `feat!: remove deprecated parameters from generator interface`
- `fix: correct Hurst exponent calculation

BREAKING CHANGE: Hurst exponent now returns float instead of tuple`

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
