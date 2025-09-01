# StatTools Technical Context

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary implementation language
- **NumPy**: Core numerical computing library
- **C++**: Performance-critical components and extensions
- **CMake**: Build system for C++ components

### Development Tools
- **Setuptools**: Python package management and building
- **pytest**: Testing framework
- **GitHub Actions**: CI/CD pipeline
- **pre-commit**: Code quality enforcement
- **setuptools-scm**: Version management

### Dependencies
- **numpy>=1.22.4**: Core numerical operations
- **pandas>=2.0.0**: Data manipulation and analysis
- **scipy**: Additional scientific computing functions
- **matplotlib**: Visualization (for examples and research)
- **tqdm**: Progress bars for long-running operations

## Development Setup

### Local Development
```bash
# Clone repository
git clone https://github.com/Digiratory/StatTools.git
cd StatTools

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Unix

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

### Build System
- **setuptools** with **setuptools-scm** for version management
- **pyproject.toml** for modern Python packaging
- **CMake** for building C++ extensions
- **Git tags** with "release/X.Y.Z" pattern for versioning

## Technical Constraints

### Performance Requirements
- **Large datasets**: Support for datasets with 10^6+ data points
- **Parallel processing**: Multi-threaded implementations for CPU-bound tasks
- **Memory efficiency**: Shared memory buffers for large matrix operations
- **Real-time processing**: Low-latency analysis for interactive research

### Compatibility Constraints
- **Python versions**: Support Python 3.8+ (follows scientific Python ecosystem)
- **Platform support**: Windows, macOS, Linux
- **NumPy compatibility**: Must work with current and recent NumPy versions
- **Thread safety**: Parallel operations must be thread-safe

### Algorithmic Constraints
- **Numerical stability**: Algorithms must be numerically stable across parameter ranges
- **Precision requirements**: Results must match theoretical expectations within acceptable error bounds
- **Edge case handling**: Robust handling of edge cases and invalid inputs

## Architecture Decisions

### Modular Design
- **Separation of concerns**: Generators, analysis, and utilities in separate modules
- **Extensible interfaces**: Easy to add new generators and analysis methods
- **Plugin architecture**: Support for custom analysis methods

### Performance Optimizations
- **C++ extensions**: Performance-critical code in C++ with Python bindings
- **Vectorized operations**: NumPy-based implementations where possible
- **Parallel processing**: Multi-threading for CPU-bound operations
- **Memory mapping**: Efficient handling of large datasets

### Code Quality
- **Type hints**: Full type annotation for better IDE support
- **Comprehensive testing**: High test coverage for critical algorithms
- **Documentation**: Sphinx-based documentation with examples
- **Code formatting**: Consistent code style with black/isort

## Development Workflow

### Version Control
- **Git Flow**: Feature branches, develop/main branches
- **Conventional commits**: Structured commit messages following Conventional Commits 1.0.0 specification. Commits must be prefixed with types like 'feat', 'fix', 'docs', etc., followed by optional scope and required description. Breaking changes indicated by '!' prefix or BREAKING CHANGE footer. Enables automated changelog generation and semantic versioning.
- **Pull requests**: Code review process for all changes
- **Release tags**: Versioned releases with changelogs

### Testing Strategy
- **Unit tests**: Individual function and method testing
- **Integration tests**: End-to-end workflow testing
- **Performance tests**: Benchmarking for performance regression detection
- **Statistical validation**: Tests against known theoretical results

### CI/CD Pipeline
- **Automated testing**: Run tests on multiple Python versions and platforms
- **Code quality checks**: Linting, formatting, and type checking
- **Documentation building**: Automatic documentation generation
- **Package building**: Automated PyPI package creation

## Tool Usage Patterns

### Core Development Tools
- **VS Code**: Primary IDE with Python extensions
- **Jupyter**: Research and prototyping
- **Git**: Version control with GitHub
- **Docker**: Containerized testing and deployment

### Research Tools
- **Jupyter notebooks**: Research documentation and examples
- **matplotlib/seaborn**: Data visualization
- **pandas**: Data manipulation for research workflows

### Performance Analysis
- **cProfile**: Python code profiling
- **line_profiler**: Line-by-line performance analysis
- **memory_profiler**: Memory usage analysis
- **pytest-benchmark**: Performance benchmarking

## Deployment and Distribution

### Package Distribution
- **PyPI**: Primary distribution channel
- **GitHub releases**: Source code and binary distributions

### Documentation
- **Read the Docs**: Hosted documentation
- **Jupyter notebooks**: Interactive examples
- **API reference**: Auto-generated from docstrings

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community questions and discussions
- **Contributing guidelines**: Clear process for external contributions
