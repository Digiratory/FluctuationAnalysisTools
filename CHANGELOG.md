# Changelog

## 1.10.0

* [GH-73](https://github.com/Digiratory/FluctuationAnalysisTools/issues/73) refactor: Restructure C++ codebase and add StatTools package with pybind11 bindings.
* [GH-71](https://github.com/Digiratory/FluctuationAnalysisTools/issues/71) cicd: Add pre-commit validation proir tests.
* [GH-67](https://github.com/Digiratory/FluctuationAnalysisTools/issues/67) feat: NEW Method multidimantional DFA.
* [GH-63](https://github.com/Digiratory/FluctuationAnalysisTools/issues/63)[PR-46](https://github.com/Digiratory/FluctuationAnalysisTools/pull/46) feat: fluctuation function analysis and visualization.
* [GH-61](https://github.com/Digiratory/FluctuationAnalysisTools/issues/61) feat: Function-style fbn generator (common entry point).
* [GH-58](https://github.com/Digiratory/FluctuationAnalysisTools/issues/58) feat: NEW Method to generate Power-law colored noise signals of any dimensions.
* [GH-50](https://github.com/Digiratory/FluctuationAnalysisTools/issues/50) refactor: Extract the core of dpcca in a separate method.
* [PR-69](https://github.com/Digiratory/FluctuationAnalysisTools/pull/69) refactor: Refactor DFA implementation into modular functions, add function style interface for dfa.
* [PR-65](https://github.com/Digiratory/FluctuationAnalysisTools/pull/65) feat: NEW Method SVD-DFA.
* [PR-60](https://github.com/Digiratory/FluctuationAnalysisTools/pull/60)[PR-78](https://github.com/Digiratory/FluctuationAnalysisTools/pull/78)[PR-80](https://github.com/Digiratory/FluctuationAnalysisTools/pull/80)[PR-81](https://github.com/Digiratory/FluctuationAnalysisTools/pull/81)[PR-82](https://github.com/Digiratory/FluctuationAnalysisTools/pull/82)[PR-83](https://github.com/Digiratory/FluctuationAnalysisTools/pull/83)[PR-85](https://github.com/Digiratory/FluctuationAnalysisTools/pull/85): Multiple minor fixes in CI/CD and code.

## 1.9.0

* [GH-13](https://github.com/Digiratory/StatTools/issues/13) fix: possible unbalances tuple unpacking from method dpcca.
* [GH-13](https://github.com/Digiratory/StatTools/issues/13) docs: update documentation for dpcca method.
* [GH-13](https://github.com/Digiratory/StatTools/issues/13) refactor: update code for dpcca method to improve performance and readability.
* [GH-18](https://github.com/Digiratory/StatTools/issues/18) feat: Enhanced Kalman filter with auto calculation of transition matrix and measurement covariance matrix based on Kasdin model.
* [GH-18](https://github.com/Digiratory/StatTools/issues/18) fix: Add normalization to the Kasdin generator.

## 1.8.0

* [GH-9](https://github.com/Digiratory/StatTools/issues/9) repo: setup pre-commit hooks.
* [GH-12](https://github.com/Digiratory/StatTools/issues/12) docs: fix format violation in CHANGELOG.md.
* [GH-15](https://github.com/Digiratory/StatTools/issues/15) feat&fix: LBFBm generator update: generate with input value and return an increment instead of the absolute value of the signal.
* [GH-23](https://github.com/Digiratory/StatTools/issues/23) feat: add Kasdin generator. fix: change first arg in lfilter in LBFBm generator.
* [GH-25](https://github.com/Digiratory/StatTools/issues/25) feat: Detrended Fluctuation Analysis (DFA) for a nonequidistant dataset.
* [GH-28](https://github.com/Digiratory/StatTools/issues/28) repo: Exclude Jupyter Notebooks from GitHub Programming Language Stats.

## 1.7.0

* [GH-5](https://github.com/Digiratory/StatTools/issues/5) feat: add LBFBm generator, that generates a sequence based on the Hurst exponent.
* [PR-8](https://github.com/Digiratory/StatTools/pull/8) refactor: rework filter-based generator.
* [PR-8](https://github.com/Digiratory/StatTools/pull/8) tests: add new tests for DFA and generators.
* [GH-10](https://github.com/Digiratory/StatTools/issues/10) build: enable wheel building with setuptools-scm.
* [GH-10](https://github.com/Digiratory/StatTools/issues/10) doc: enchance pyproject.toml with urls for repository, issues, and changelog.

## 1.6.1

* [PR-3](https://github.com/Digiratory/StatTools/pull/3) feat: add conventional FA

## 1.6.0

* [GH-1](https://github.com/Digiratory/StatTools/issues/1) Add argument `n_integral=1` in `StatTools.analysis.dpcca.dpcca` to provide possibility to control integretion in the beggining of the dpcca(dfa) analysis pipeline.
* fix: failure is processes == 1 and 1d array
* fix: remove normalization from dpcca processing

## 1.0.1 - 1.0.9

* Minor updates

## 1.1.0

* Added C-compiled modules
