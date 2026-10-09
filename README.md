# Macro-F1 prevalence frontiers

Reproducibility code and derived results for *Single-Crossing Macro-F1 Frontiers and Finite-Sample Certificates under Prevalence Shift*.

The repository contains the finite-policy frontier algorithm, simultaneous rate bounds, simulations, aggregate application results, and figure generators. The complete numerical CSVs, including negative results, are retained. It contains no individual prediction records or original task data.

## Quick verification

Use Python 3.12.10. The recorded scientific package versions are pinned in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python innovation/test_scientific_properties.py
python innovation/verify_results.py
```

The first command after installation runs nine scientific property tests. The result checker verifies file checksums, recomputes summaries and Monte Carlo standard errors, checks the calibration-draw count, and recomputes archived application certificates and continuum regret bounds from aggregate conditional rates. It does not rerun training or Monte Carlo experiments. These checks support consistency of the supplied implementation and results; they do not replace the mathematical proof or establish generalization of the real-task descriptions.

## Rebuild figures from existing results

```sh
python innovation/plot_frontier.py
python innovation/plot_results.py
python innovation/plot_extended_experiments.py
python innovation/plot_runtime.py
```

Figures are written to `innovation/figures/` as PNG, PDF and SVG. Plot data come from the frozen CSV/JSON tables, without new calibration samples. The runtime plot uses recorded timings; timings from a fresh run will depend on hardware and software.

## Materials

| File or directory | Purpose |
| --- | --- |
| `innovation/policy_certificate.py` | Macro-F1, pair crossings, all-pairs reference, Pareto stack frontier, CP/DKW bounds, policy/learner certificates, interval regret bound |
| `innovation/run_development.py` | Original simulation and archive-backed application analysis |
| `innovation/extended_experiments.py` | Sample-size, imbalance, confidence-level, dependence and prevalence-grid sensitivity |
| `innovation/extended_archive_and_pool.py` | Threshold-pool sensitivity and archive resplitting |
| `innovation/orthogonal_pool_ablation.py` | Controlled frontier-density and pool-size sensitivity with DKW M=K |
| `innovation/benchmark_runtime.py` | Complete-function stack versus all-pairs runtime comparison |
| `innovation/results/` | Frozen simulation records, summaries, metadata, aggregate task rates and phase diagrams |
| `innovation/results/diagnostics/` | Aggregate zero-certificate diagnostics and summaries |
| `innovation/results/runtime/` | Recorded timing repetitions, summaries and design metadata |
| `innovation/results/source_tasks.json` | Original task identifiers, class definitions, dataset links and recorded snapshot hashes |
| `innovation/results/SHA256.json` | Integrity manifest for the supplied derived result files |

## Design and interpretation

The guarantees concern frozen finite binary policies under pure prevalence (label) shift, with class-conditional IID calibration independent of model fitting. Clopper-Pearson (CP) additionally requires a candidate pool fixed independently of calibration labels. The DKW allocation counts score functions rather than thresholds; the controlled binary-column experiment uses one score function per policy, M=K. A policy certificate distinguishes individual policies; a learner certificate compares learner groups.

The frontier uses numerical root solving with a high-precision fallback. The continuum regret bound adds an explicit Lipschitz correction to a prevalence grid. These implementations use floating-point arithmetic, not interval arithmetic certification.

There are 71,900 simulated calibration draws. Paired bands, candidate-pool views and reused prevalence grids are counted once per draw:

| Experiment | Calibration draws |
| --- | ---: |
| Original simulation | 12,000 |
| Broad sample-size and imbalance sensitivity | 33,600 |
| Confidence-level sensitivity | 3,000 |
| Dependence sensitivity | 2,400 |
| Grid-resolution sensitivity | 400 |
| Threshold-pool sensitivity | 4,500 |
| Controlled pool sensitivity | 16,000 |
| Total | 71,900 |

Each script contains its recorded seed rule and design. CP and DKW share observations within a condition and repetition; controlled-pool conditions use different seeds. Repeated grids and forest-seed views are not independent experimental units.

The archived analysis has nine tasks and 135 outer-fold blocks; forest seed 101 is the primary view. Of these blocks, 102 exhibit a descriptive policy switch. Strict policy and learner certification are zero at the three evaluated prevalences. This does not establish that the learners have equal population performance. Zero-certificate diagnostics decompose observed gaps and uncertainty penalties; they do not by themselves identify causal effects of sample size or candidate density.

## Optional full simulation runs

Run these only in a separate copy: they overwrite the corresponding frozen result tables and change the checksum manifest's validity.

```sh
python innovation/run_development.py simulation --reps 1000
python innovation/extended_experiments.py
python innovation/extended_archive_and_pool.py pool
python innovation/orthogonal_pool_ablation.py
python innovation/benchmark_runtime.py
```

The runtime comparison uses two deterministic rate families, three timing repetitions and K in {32, 64, 128, 256, 1024}. All-pairs timing stops at K=256. Both functions share the pair-root helper; timing agreement is not an independent root-accuracy proof. Root counts do not describe every operation in the complete implementation. No speed claim is extrapolated beyond measured cases.

## Real-task reproduction and data access

Public files allow recomputation of application certificates from aggregate class-conditional rates, reaggregation of resplit results and reconstruction of the supplied plots. They do **not** allow regeneration of individual prediction archives, retraining of the original fitted models, or reconstruction of the resplit observations.

`source_tasks.json` identifies the OpenML datasets and recorded snapshots used by the underlying archive. Follow the source dataset links for the applicable terms and availability; this repository does not redistribute source records. It contains no original training pipeline or individual-level archives. No public access entitlement to those archives is asserted here. Any additional archive access requires separate authorization from its holder; the public checks and plots do not require it.

The two archive-backed entry points are provided for readers who already possess an independently authorized, compatible archive:

```sh
python innovation/run_development.py application
python innovation/extended_archive_and_pool.py archive-resplit
```

They read `CISSC_ARCHIVE_ROOT`, an environment variable pointing to that archive. The expected schema is `openml_source_schema_audit.json` with a `datasets` array containing `name`, `n` and `positive_n`; each block in `results/` supplies `outer_predictions.parquet`, `metrics.csv`, and (for resplitting) `complete.json`. Prediction columns and their use are explicit in the scripts. Parquet reading additionally needs `pyarrow`, whose original version was not recorded in the supplied environment file. Archive-backed regeneration is outside the public-only reproducibility scope.

## License

The repository retains its MIT license. This license does not change the access terms or licenses of external source datasets.
