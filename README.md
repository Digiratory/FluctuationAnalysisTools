# FluctuationAnalysisTools

[![GitHub Release](https://img.shields.io/github/v/release/Digiratory/StatTools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)](https://pypi.org/project/FluctuationAnalysisTools/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Digiratory/StatTools/run-tests.yml?label=tests)](https://github.com/Digiratory/StatTools/actions)
[![GitHub License](https://img.shields.io/github/license/Digiratory/StatTools)](https://github.com/Digiratory/StatTools/blob/main/LICENSE.txt)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fluctuationanalysistools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)](https://pypi.org/project/FluctuationAnalysisTools/)

A Python library for creating and processing long-term dependent datasets, with a focus on statistical analysis tools for fluctuation analysis, time series generation, and signal processing.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Research and Examples](#research-and-examples)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Features

- **Synthetic Data Generation**: Create datasets with controlled statistical properties (Hurst exponent, long-term dependencies)
- **Fluctuation Analysis**: Perform Detrended Fluctuation Analysis (DFA), Detrended Partial Cross-Correlation Analysis (DPCCA), and other methods
- **Signal Processing**: Apply filters and transformations to time series data
- **Research Tools**: Support scientific research in complex systems exhibiting long-range correlations
- **Performance Optimized**: Multi-threaded implementations with C++ extensions for large datasets

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for visualization)
- C++ compiler (Mac and Windows packages contains precompiled binaries)

## Installation

You can install FluctuationAnalysisTools from [PyPI](https://pypi.org/project/FluctuationAnalysisTools/):

```bash
pip install FluctuationAnalysisTools
```

Or clone the repository and install locally:

```bash
git clone https://github.com/Digiratory/FluctuationAnalysisTools.git
cd StatTools
pip install .
```

## Quick Start

You can find examples and published usages in the folder [Research](./research/)

If you used the project in your paper, you are welcome to ask us to add reference via a Pull Request or an Issue.

### Generating Synthetic Data

```python
from StatTools.generators import generate_fbn
import numpy as np

# Create a dataset with Hurst exponent H = 0.8 using the unified interface
hurst = 0.8
length = 1440

# Generate fractional Brownian noise using the default Kasdin method
fbn_series = generate_fbn(hurst=hurst, length=length)
print(f"Generated fBn series with shape: {fbn_series.shape}")
```

### Analyzing Time Series

#### Detrended Fluctuation Analysis

```python
from StatTools.generators import generate_fbn
from StatTools.analysis.dfa import dfa
from StatTools.analysis.utils import analyse_zero_cross_ff
import numpy as np

h = 0.7  # choose Hurst parameter
length = 6000  # vector's length

# Generate synthetic data using modern unified interface
trajectory = generate_fbn(hurst=h, length=length, method="kasdin").flatten()

# Analyze using functional-style DFA interface
s_vals, f2_vals = dfa(trajectory, degree=2)

# Calculate Hurst exponent from fluctuation function
f_vals = np.sqrt(f2_vals).reshape(1, -1)  # Convert F^2(s) to F(s) and reshape for analysis
s_vals_2d = s_vals.reshape(1, -1)  # Reshape scales to 2D array
hurst_result, _ = analyse_zero_cross_ff(f_vals, s_vals_2d)
hurst_exponent = hurst_result.slopes[0].value

print(f"Estimated H: {hurst_exponent:.3f} (Expected: {h:.3f})")
```

## Support

For questions and discussions:

- GitHub Issues: <https://github.com/Digiratory/FluctuationAnalysisTools/issues>
- GitHub Discussions: <https://github.com/Digiratory/FluctuationAnalysisTools/discussions>

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

## License

This project is licensed under the terms specified in [LICENSE.txt](LICENSE.txt).

## Citation

If you use FluctuationAnalysisTools in your research, please cite:

```bibtex
@article{bogachev2023understanding,
  title={Understanding the complex interplay of persistent and antipersistent regimes in animal movement trajectories as a prominent characteristic of their behavioral pattern profiles: Towards an automated and robust model based quantification of anxiety test data},
  author={Bogachev, Mikhail I and Lyanova, Asya I and Sinitca, Aleksandr M and Pyko, Svetlana A and Pyko, Nikita S and Kuzmenko, Alexander V and Romanov, Sergey A and Brikova, Olga I and Tsygankova, Margarita and Ivkin, Dmitry Y and others},
  journal={Biomedical signal processing and control},
  volume={81},
  pages={104409},
  year={2023},
  publisher={Elsevier}
}
```

and

```bibtex
@article{bogachev2023video,
  title={Video-based marker-free tracking and multi-scale analysis of mouse locomotor activity and behavioral aspects in an open field arena: a perspective approach to the quantification of complex gait disturbances associated with Alzheimer's disease},
  author={Bogachev, Mikhail and Sinitca, Aleksandr and Grigarevichius, Konstantin and Pyko, Nikita and Lyanova, Asya and Tsygankova, Margarita and Davletshin, Eldar and Petrov, Konstantin and Ageeva, Tatyana and Pyko, Svetlana and others},
  journal={Frontiers in Neuroinformatics},
  volume={17},
  pages={1101112},
  year={2023},
  publisher={Frontiers Media SA}
}
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
