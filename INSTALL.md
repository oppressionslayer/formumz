# Install / Setup Notes

The demos are self-contained Python scripts. Install the libraries with pip,
then run any script directly.

## Requirements

- **Python 3.10 or newer.** Tested on 3.13.
- `numpy >= 1.22`
- `scikit-learn >= 1.2`
- `matplotlib >= 3.5` (only for `02_parity_full.py` chart generation)
- `xgboost` (optional; the full benchmark script gracefully skips it if missing)

```bash
pip install numpy scikit-learn matplotlib xgboost
```

## The libraries

The demos import three libraries. Install them with pip:

```bash
pip install wsrf-lib lars-logic schur-logic
```

| Demo | Library used |
|---|---|
| `01_parity_90s.py` | `wsrf-lib` |
| `02_parity_full.py` | `wsrf-lib` |
| `03_rule30.py` | `schur-logic` |
| `04_syndrome.py` | `lars-logic` |

## Running the demos

From this folder:

```bash
python demos/01_parity_90s.py
python demos/02_parity_full.py
python demos/03_rule30.py
python demos/04_syndrome.py
```

## Known quirk: macOS Python 3.13 + joblib

On Apple Silicon Macs running Python 3.13 with parallel-fitting scikit-learn
models, you'll see harmless stderr noise like:

```
ValueError: Cannot register /loky-XXXXX-yyyy for automatic cleanup:
unknown resource type semlock
```

This comes from `joblib`/`loky`'s resource-tracker being stricter on
Python 3.13. The actual fits succeed; only the cleanup process trips. To
suppress the noise:

```bash
python demos/01_parity_90s.py 2>/dev/null
```

`2>/dev/null` discards stderr; the headline results print to stdout regardless.

## Approximate runtimes (Apple Silicon M-series, 2026)

| Demo | Runtime |
|---|---|
| `01_parity_90s.py` | ~90 seconds |
| `02_parity_full.py` | ~20–30 minutes (4 phases × 5 seeds × 7 models) |
| `03_rule30.py` | ~30 seconds (mostly Step 5 scale timing) |
| `04_syndrome.py` | ~5 seconds |

Older or slower machines scale roughly proportionally; the work is mostly
sklearn-bound for the parity benchmarks and pure-Python for everything else.

## Output locations

Demos write to:

- `02_parity_full.py` → `wsrf_benchmark_out/` (auto-created next to the script)
- `04_syndrome.py` → stdout only
- All others → stdout only

The `results/` folder in this repo already contains pre-computed artifacts
(`REPORT.md`, `accuracy.png`, `noise.png`, `results.csv`) so a reader doesn't
need to re-run the 30-minute benchmark to see the numbers.

## If something breaks

1. **`ModuleNotFoundError: No module named 'wsrf'` (or `lars_logic` / `schur_logic`):**
   The library isn't installed in the active Python environment. Run
   `pip install wsrf-lib lars-logic schur-logic` and try again.

2. **xgboost not installed:** The full benchmark gracefully skips it. The
   90-second demo will fail to instantiate it — `pip install xgboost` or edit
   the script to comment out the XGBoost line.

3. **Stderr spam on Python 3.13:** see the joblib note above. Use `2>/dev/null`.

4. **Different numbers than the README table:** report it. The benchmarks are
   seeded; CI bounds should be tight. Large drift would be interesting to
   debug.

Open an issue if reproduction fails — replication reports are the kind of
thing that actually matter here.
