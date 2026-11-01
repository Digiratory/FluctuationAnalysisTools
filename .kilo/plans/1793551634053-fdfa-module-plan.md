# FDFA — Functional Detrended Fluctuation Analysis (GH-136)

## Context

- Issue: https://github.com/Digiratory/FluctuationAnalysisTools/issues/136 — port **FDFA**, a two-layer
  generalization of (MF)DFA where the local fluctuation measure is a pluggable dispersion functional Φ
  acting on detrending residuals: `F_Φ(v, s) = Φ(r_1..r_s)`. Classical DFA/MFDFA is recovered with Φ = RMS.
- Reference implementation (MIT, NumPy-only core, Python ≥3.10 syntax):
  https://github.com/sierraporta/functional-dfa/blob/v2.0.0/fdfa.py — full source already retrieved and
  reviewed; port it, adapted to repo conventions.
- Repo: `StatTools/analysis/` holds functional-style methods (`dfa()`, `fa()`, `svd_dfa()`, `dpcca()`).
  Docstrings are Google style (`Args:` / `Returns:`). CI: Python 3.11, black + isort (black profile) via
  pre-commit. `dfa.py` already uses `int | list` annotations, so modern syntax is fine.
- **Decision (user-confirmed):** keep the reference window construction — forward + backward passes
  (standard DFA/Peng convention). Existing repo `dfa()` uses forward-only windows, so Φ=RMS reproduces
  existing DFA **slopes within tolerance**, not bit-identical F² values.

## Locked design decisions

1. New module `StatTools/analysis/fdfa.py`; public API re-exported from `StatTools/analysis/__init__.py`.
2. Port the reference **as-is** algorithmically (windowing, Gaussian-consistency constants, IRLS robust
   detrending, qmean/median aggregation, lstsq fit with stderr). Only style/API-convention adaptations.
3. Keep the core NumPy-only (the reference's private `_norm_ppf` bisection, gamma constants) — no scipy
   in the module core. scipy is allowed in tests.
4. Keep reference names, including the `hurst()` convenience helper (no name collision in the package).
5. `fdfa` is 1-D input only; silently drops non-finite samples (reference behavior, documented).
   `*_multi` variants are multi-*functional* (shared residuals per scale), not multi-series.
6. Attribution: MIT, module docstring credits D. Sierra-Porta's framework + reference repo/zenodo links.

## Tasks (ordered)

### 1. `StatTools/analysis/fdfa.py`

Port from reference, restructured in this order:

- Gaussian-consistency constants: `_SQRT2`, `_phi_pdf`, `_norm_cdf`, `_norm_ppf` (bisection),
  `_lp_gauss_const`, `_MAD_CONST`, `_ENTROPY_GAUSS`.
- `Functional` frozen dataclass: `name`, `fn` ((n_windows, s) → (n_windows,)), `moment_requirement`,
  `breakdown`; `__call__` normalizes to 2-D float.
- Built-in battery (all σ-consistent under Gaussian residuals):
  `rms`, `make_lp(p)`, `l1`, `mad`, `make_quantile(tau)`, `iqr`, `make_huber(c, n_iter)` (Proposal-2 M-estimator),
  `linf`, `make_trimmed_std(alpha)`, `entropy_power` (Vasicek spacings), `FUNCTIONALS` dict (same 11 entries
  and keys as reference: rms, l1, lp0.5, mad, q0.75, q0.9, iqr, huber, tstd10, linf, entpow).
- `default_scales(n, s_min=16, s_max=None, n_scales=20)` — log-spaced unique ints, default s_max = n//4.
- Core internals with `functools.lru_cache`: `_design_matrix(s, order)` (standardized abscissa),
  `_hat_matrix(s, order)` (QR projection); `_robust_residuals` (IRLS, loss in {'l1','huber'}, vectorized);
  `_windows(Y, s, overlap)` (overlap=0 → fwd + bwd passes; else step `max(1, round((1-overlap)*s))`);
  `_window_residuals`; `_aggregate` (qmean with |q|<1e-12 → geometric mean; median; else ValueError).
- Result dataclasses: `FDFAResult(scales, F, slope, intercept, stderr, functional, order, aggregate, Fv)`,
  `HqResult(qs, hq, tq, scales, Fq, functional)`.
- Public API:
  - `fdfa(x, phi=rms, *, scales=None, order=1, aggregate="qmean", q=2.0, fit_range=None, detrend="ols",
    overlap=0.0, keep_windows=False)` — profile `Y = cumsum(x - mean)`, scales filtered to
    `[order+2, n//2]`, per-scale Φ on residuals, aggregation, OLS log–log fit (slope/intercept/stderr).
    Raise `ValueError` if no valid scales/fit points remain (repo-style validation; the reference would
    return NaNs — deviate deliberately and raise).
  - `hurst(x, phi=rms, **kw)` — slope only.
  - `fdfa_multi(x, specs, ...)` — specs = iterable of `(label, Functional, aggregate, q)`; residuals
    computed once per scale, shared; stderr left NaN as in reference.
  - `mfdfa_hq(x, qs, phi=rms, ...)` → `HqResult` with `tq = q·h(q) − 1`.
  - `mfdfa_hq_multi(x, qs, phis, ...)` — dict label→HqResult sharing residuals.
- Conventions: `from __future__ import annotations`, Google-style docstrings with a short usage example in
  the module docstring, no `Dict`/`Sequence` from `typing` (use `collections.abc`), black/isort clean.
- Perf note: in `_robust_residuals` avoid `X[None].repeat(N, 0)`; broadcast X inside `einsum` directly.

### 2. `StatTools/analysis/__init__.py`

Re-export: `Functional`, `FDFAResult`, `HqResult`, `FUNCTIONALS`, `default_scales`, `fdfa`, `fdfa_multi`,
`mfdfa_hq`, `mfdfa_hq_multi`, `hurst`, and built-ins `rms`, `l1`, `mad`, `iqr`, `linf`, `entropy_power`,
`make_lp`, `make_quantile`, `make_huber`, `make_trimmed_std`. Keep import style consistent with existing
`from .bma import bma` lines.

### 3. `tests/test_fdfa.py`

Follow repo test patterns: fixed seeds, `@pytest.mark.timeout(300)` on slow statistical tests,
`IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"` guards + reduced load in CI
(see `tests/test_dfa_func.py`, `tests/test_dpcca.py`).

1. **Exact recovery (acceptance #1):**
   - Φ=RMS, q=2, `detrend="ols"`, overlap=0 equals a hand-computed classical DFA (fwd+bwd windows,
     OLS order-m residuals, `sqrt(mean r²)` aggregation) exactly per scale.
   - Slope agreement with repo `dfa()` (StatTools.analysis.dfa): `fdfa(order=1).slope` vs
     `dfa(degree=1)` slope on the same fGn series, `rel=0.1` (window sets differ: fwd+bwd vs fwd-only).
2. **Known-H recovery:** fGn from `StatTools.generators.generate_fbn` (kasdin), H ∈ {0.3, 0.5, 0.75, 1.0},
   length ~2**13, `fdfa` with `rms`, `rel=0.15` (matches existing test tolerances).
3. **Gaussian consistency (acceptance #2):** on one seeded fGn series, all `FUNCTIONALS` battery slopes
   pairwise agree within `abs≈0.15`.
4. **α-stable / heavy tails (acceptance #3):**
   - Cauchy (α=1, `np.random.standard_cauchy`, seeded, length ~2**14): `rms` slope ≈ 0.5 (RMS-DFA blind),
     robust Φ (`mad`, `huber`) slope ≈ 1/α = 1, tolerances wide (e.g. rms ∈ [0.35, 0.65], mad ∈ [0.7, 1.3]).
   - One more α via `scipy.stats.levy_stable.rvs(alpha, beta=0)` (e.g. α=1.5 → H≈0.667): robust Φ recovers
     1/α loosely; guard with `IN_GITHUB_ACTIONS` if slow.
5. **Multifractal layer:** for monofractal fGn, `mfdfa_hq` gives h(q) ≈ H (flat in q) and τ(q) linear in q
   with slope H and intercept −1; for `rms`/q=2 the h(q) at q=2 equals the `fdfa` slope.
6. **Batch API:** `fdfa_multi` results match individual `fdfa` calls (F arrays allclose);
   `mfdfa_hq_multi` matches `mfdfa_hq`.
7. **API/edge cases:** invalid `aggregate` raises ValueError; empty/all-invalid `scales` raises ValueError;
   scales < order+2 filtered; `fit_range` restricts fit; `keep_windows` populates `Fv`; `overlap=0.5` runs;
   robust `detrend="l1"/"huber"` run and return finite slopes; `hurst()` returns float; input validation for
   non-1-D input raises.

### 4. Documentation

- `README.md`: add FDFA bullet under **Features → Fluctuation Analysis**; add a short subsection under
  **Analyzing Time Series** with a minimal example (`fdfa` with `rms` and with `mad`, noting the
  α-stable motivation) and citation of Sierra-Porta's paper + zenodo DOI 10.5281/zenodo.22769472.
- `CHANGELOG.md`: new top section (follow existing heading style, e.g. `## Unreleased`) with
  `[GH-136](...) feat: Functional detrended fluctuation analysis (FDFA).`

### 5. Validation

```bash
python -m pytest tests/test_fdfa.py -v          # new suite
python -m pytest tests/                          # full suite, no regressions (target tests/, never bare pytest)
pre-commit run --files StatTools/analysis/fdfa.py StatTools/analysis/__init__.py tests/test_fdfa.py README.md CHANGELOG.md
```

Statistical tolerances are seed-fixed; if a known-H test flakes, widen tolerance rather than tune the
algorithm (repo convention: occasional flakiness from statistical tolerances is expected).

## Risks / notes

- Φ=RMS F values intentionally differ from repo `dfa()` F² (fwd+bwd vs fwd-only windows) — slopes must
  match; do not "fix" by switching to fwd-only.
- `linf` and `entropy_power` have no finite Gaussian normalization constants (`linf` grows ~√(2 log s));
  on Gaussian data they still give approximately consistent slopes — that is what test 3 checks.
- `scipy.stats.levy_stable` generation can be slow for large n; keep n modest (~2**14) and timeout-mark it.
- IRLS robust detrending is the most numerically delicate part; the reference's ridge `+eps·I` and MAD
  fallbacks must be preserved.

## Out of scope (separate issues, per GH-136)

- Multifractality-type discriminator (quantile-exponent profile θ(τ), surrogate tests).
- Heavy-tailed (α-stable) benchmark **generators** in `StatTools/generators` — tests generate α-stable
  series locally instead.
- CUDA backend, multi-series (2-D) support, parallelism for FDFA.
- C++ side: no changes; new module is pure Python/NumPy.
