# StatTools

This library allows to create and process long-term dependent datasets.

![GitHub Release](https://img.shields.io/github/v/release/Digiratory/StatTools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Digiratory/StatTools/run-tests.yml?label=tests)
![GitHub License](https://img.shields.io/github/license/Digiratory/StatTools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fluctuationanalysistools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)

## Installation

You can install FluctuationAnalysisTools from [PyPI](https://pypi.org/project/FluctuationAnalysisTools/).

```bash
pip install FluctuationAnalysisTools
```

Or you can clone the repository and build it using the command

```bash
pip install .
```

## Examples

You can find examples and published usages in the folder [Research](./research/readme.md)

If you used the project in your paper, you are welcome to ask us to add reference via a Pull Request or an Issue.

## Basis usage

1. To create a simple dataset with given Hurst parameter:

```python
from StatTools.filters import FilteredArray

h = 0.8                 # choose Hurst parameter
total_vectors = 1000    # total number of vectors in output
vectors_length = 1440   # each vector's length
t = 8                   # threads in use during computation

correlated_vectors = Filter(h, vectors_length).generate(n_vectors=total_vectors,
                                                        threads=t, progress_bar=True)
```

### Generators

1. Example of sequence generation based on the Hurst exponent.

```python
from StatTools.generators.lbfbm_generator import LBFBmGenerator

# Parameters
hurst_exponent = 1.2   # choose Hurst parameter H ∈ (0, 2)
base = 1.2             # the basis for the filter optimization algorithm
sequence_length = 4000 # Desired length of the generated signal

# Create the generator instance
generator = LBFBmGenerator(h=hurst_exponent, base=base, length=sequence_length)

# Generate the signal
signal = list(generator)
```

For more information and generator validation, see [lbfbm_generator.ipynb](/research/lbfbm_generator.ipynb).

It is also possible to use the method of generating increments with a given H using `KasdinGenerator`.

```python
from StatTools.generators.kasdin_generator import KasdinGenerator
h = 0.8             # choose Hurst parameter
target_len = 4000   # number of generation iterations

generator = KasdinGenerator(h, length=target_len)

# the first option
signal = generator.get_full_sequence()

# the second option
signal_list = []
for sample in generator:
    signal_list.append(sample)
```
For more information see Kasdin, N. J. (1995). Discrete simulation of colored noise and stochastic processes and 1/f/sup /spl alpha// power law noise generation. doi:10.1109/5.381848.

### Fluctuational Analysis

1. Example of Detrended Fluctuational Analysis (DFA)

```python
from StatTools.generators.base_filter import Filter
from StatTools.analysis.dfa import DFA

h = 0.7 # choose Hurst parameter
length = 6000 # vector's length
target_std = 1.0
target_mean = 0.0

generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
trajectory = generator.generate(n_vectors=1)

actual_mean = np.mean(trajectory)
actual_std = np.std(trajectory, ddof=1)
actual_h = DFA(trajectory).find_h()
print(actual_h) # Should print a value close to 0.7
```

## Contributors

* [Alexandr Kuzmenko](https://github.com/alexandr-1k)
* [Aleksandr Sinitca](https://github.com/Sinitca-Aleksandr)
* [Asya Lyanova](https://github.com/pipipyau)
* [Maya Gavrichenkova](https://github.com/MayaGavrichenkova)
