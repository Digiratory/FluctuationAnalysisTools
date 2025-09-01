# StatTools

[![GitHub Release](https://img.shields.io/github/v/release/Digiratory/StatTools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)](https://pypi.org/project/FluctuationAnalysisTools/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Digiratory/StatTools/run-tests.yml?label=tests)](https://github.com/Digiratory/StatTools/actions)
[![GitHub License](https://img.shields.io/github/license/Digiratory/StatTools)](https://github.com/Digiratory/StatTools/blob/main/LICENSE.txt)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fluctuationanalysistools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)](https://pypi.org/project/FluctuationAnalysisTools/)

A Python library for creating and processing long-term dependent datasets, with a focus on statistical analysis tools for fluctuation analysis, time series generation, and signal processing.

## Features

- **Synthetic Data Generation**: Create datasets with controlled statistical properties (Hurst exponent, long-term dependencies)
- **Fluctuation Analysis**: Perform Detrended Fluctuation Analysis (DFA), Detrended Partial Cross-Correlation Analysis (DPCCA), and other methods
- **Signal Processing**: Apply filters and transformations to time series data
- **Research Tools**: Support scientific research in complex systems exhibiting long-range correlations
- **Performance Optimized**: Multi-threaded implementations with C++ extensions for large datasets

## Installation

You can install StatTools from [PyPI](https://pypi.org/project/FluctuationAnalysisTools/):

```bash
pip install FluctuationAnalysisTools
```

Or clone the repository and install locally:

```bash
git clone https://github.com/Digiratory/StatTools.git
cd StatTools
pip install .
```

## Quick Start

You can find examples and published usages in the folder [Research](./research/readme.md)

If you used the project in your paper, you are welcome to ask us to add reference via a Pull Request or an Issue.


### Generating Synthetic Data

```python
from StatTools.generators.base_filter import FilteredArray
import numpy as np

# Create a dataset with Hurst exponent H = 0.8
h = 0.8
total_vectors = 1000
vectors_length = 1440
threads = 8

# Generate correlated vectors
generator = FilteredArray(h, vectors_length)
correlated_vectors = generator.generate(n_vectors=total_vectors, threads=threads, progress_bar=True)

print(f"Generated {len(correlated_vectors)} vectors of length {len(correlated_vectors[0])}")
```

### Analyzing Time Series

```python
from StatTools.analysis.dfa import DFA
import numpy as np

# Generate sample data
np.random.seed(42)
data = np.random.randn(10000)

# Perform Detrended Fluctuation Analysis
dfa = DFA(data)
hurst_exponent = dfa.find_h()

print(f"Estimated Hurst exponent: {hurst_exponent:.3f}")
```

## Examples

### Generators

#### Logarithmic Fractional Brownian Motion (LBFBM) Generator

```python
from StatTools.generators.lbfbm_generator import LBFBmGenerator

# Parameters
hurst_exponent = 0.8  # H ∈ (0, 2)
base = 1.2
sequence_length = 4000

# Create and use generator
generator = LBFBmGenerator(h=hurst_exponent, base=base, length=sequence_length)
signal = list(generator)

print(f"Generated signal of length {len(signal)}")
```

For more details, see [lbfbm_generator.ipynb](research/lbfbm_generator.ipynb).

#### Kasdin Generator for Colored Noise

```python
from StatTools.generators.kasdin_generator import KasdinGenerator

h = 0.8
target_len = 4000

generator = KasdinGenerator(h, length=target_len)

# Generate full sequence
signal = generator.get_full_sequence()
print(f"Generated signal: {signal[:10]}...")  # First 10 values

# Or iterate through samples
signal_list = []
for sample in generator:
    signal_list.append(sample)
```

Reference: Kasdin, N. J. (1995). Discrete simulation of colored noise and stochastic processes and 1/f^α power law noise generation. DOI:10.1109/5.381848.

### Fluctuation Analysis

#### Detrended Fluctuation Analysis (DFA)

```python
from StatTools.generators.base_filter import Filter
from StatTools.analysis.dfa import DFA
import numpy as np

h = 0.7  # choose Hurst parameter
length = 6000  # vector's length
target_std = 1.0
target_mean = 0.0

generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
trajectory = generator.generate(n_vectors=1)[0]  # Get the first (and only) trajectory

actual_mean = np.mean(trajectory)
actual_std = np.std(trajectory, ddof=1)
actual_h = DFA(trajectory).find_h()
print(f"Estimated H: {actual_h:.3f} (Expected: {h:.3f})")
```

## API Reference

### Generators
- `FilteredArray`: Base class for generating correlated datasets
- `LBFBmGenerator`: Linear Fractional Brownian Motion generator
- `KasdinGenerator`: Colored noise generator using Kasdin's method
- `FieldGenerator`: Spatial data generator

### Analysis Tools
- `DFA`: Detrended Fluctuation Analysis
- `DPCCA`: Detrended Partial Cross-Correlation Analysis
- `FA`: Fluctuation Analysis
- `QSS`: Quantile Segmentation Statistics

### Filters
- `KalmanFilter`: Kalman filtering implementation

## Research and Examples

Find comprehensive examples and published research in the [research/](research/) folder:

- [Kalman Filter Examples](research/kalman_filter.ipynb)
- [LBFBM Generator Validation](research/lbfbm_generator.ipynb)
- [Video-based Analysis](research/Video-based_marker-free_tracking_and_multi-scale_analysis.ipynb)

If you've used StatTools in your research, consider contributing your examples via a Pull Request or Issue.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTION.md) for details on:

- Setting up a development environment
- Code style and standards
- Testing guidelines
- Submitting pull requests

## Contributors

- [Alexandr Kuzmenko](https://github.com/alexandr-1k)
- [Aleksandr Sinitca](https://github.com/Sinitca-Aleksandr)
- [Asya Lyanova](https://github.com/pipipyau)

## License

This project is licensed under the terms specified in [LICENSE.txt](LICENSE.txt).

## Citation

If you use StatTools in your research, please cite:

```bibtex
@software{statttools,
  title = {StatTools: A Python Library for Long-term Dependent Dataset Analysis},
  author = {Kuzmenko, Alexandr and Sinitca, Aleksandr and Lyanova, Asya},
  url = {https://github.com/Digiratory/StatTools},
  version = {1.6.1}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
