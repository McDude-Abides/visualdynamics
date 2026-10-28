# Benchmarks for the Rust port

Two scripts, run with the repository's own virtual environment:

- `inventory.py` walks the package's AST and writes `out/inventory.csv`
  (every function and method: layer, size, loops, numpy/scipy use,
  whether its module imports Qt) and `out/modules.csv`.
- `baseline.py` times the numeric kernels across input sizes and writes
  `out/baseline-<commit>.csv` and `.json`. `--quick` runs the smaller
  sizes; names given on the command line select cases.

## Baseline

visualdynamics 0.1.0a74 at `4c9b76d`, Python 3.14.8,
numpy 2.5.4, scipy 1.18.1, 12th Gen Intel(R) Core(TM) i7-12700 (20 threads).
Minimum of repeated calls after one warm-up call. The exponent is a
log-log fit over the larger half of the sizes.

| Case | Function | Varies | Held fixed | Largest | Time at largest | Exponent |
|---|---|---|---|---:|---:|---:|
| srs.samples | `core.srs.peaks` | samples | 105 frequencies | 1,000,000 | 3,363.4 ms | 1.00 |
| srs.frequencies | `core.srs.peaks` | frequencies | 16384 samples | 3,000 | 222.7 ms | 0.42 |
| psd.samples | `TimeHistory.compute_psds` | samples | 8 channels, 1024 frame, 50% overlap | 4,194,304 | 317.7 ms | 1.04 |
| psd.channels | `TimeHistory.compute_psds` | channels | 262144 samples | 128 | 269.0 ms | 0.96 |
| cpsd.channels | `TimeHistory.compute_cpsds` | channels | 65536 samples | 64 | 597.4 ms | 1.94 |
| frf.Hv.responses | `TimeHistory.compute_frfs(Hv)` | responses | 2 references, 65536 samples | 64 | 119.2 ms | 1.08 |
| frf.H1.responses | `TimeHistory.compute_frfs(H1)` | responses | 2 references, 65536 samples | 64 | 116.3 ms | 1.04 |
| frf.Hv.samples | `TimeHistory.compute_frfs(Hv)` | samples | 8 responses, 2 references | 4,194,304 | 911.5 ms | 1.03 |
| filter.samples | `core.filters.filtered` | samples | 8 channels, low-pass 400 Hz order 4 | 4,194,304 | 129.4 ms | 0.80 |
| resample.samples | `core.resample.resample` | samples | 8 channels, rate / 4 | 4,194,304 | 267.1 ms | 0.83 |
| scalogram.samples | `core.wavelet.scalogram` | samples | 5-1000 Hz | 1,048,576 | 1,298.5 ms | 1.01 |
| scalogram_peaks.samples | `core.wavelet.scalogram_peaks` | samples | 5-1000 Hz | 4,194,304 | 7,166.9 ms | 1.05 |
| kurtosis.frames | `core.kurtosis.frame_kurtosis` | frames | 1024 per frame | 16,384 | 262.0 ms | 1.08 |
| time_statistics.samples | `core.time_statistics.time_statistics` | samples | defaults | 4,194,304 | 314.5 ms | 1.11 |
| mac.modes | `core.shapes.mac_matrix` | modes | 3000 DOFs | 1,000 | 18.5 ms | 1.30 |
| mac.dofs | `core.shapes.mac_matrix` | dofs | 50 modes | 100,000 | 19.2 ms | 1.06 |
| synthesize_frf.pairs | `ShapeSet.synthesize_frf` | pairs | 40 modes, 2048 lines | 1,000 | 15.4 ms | 0.65 |
| octave.lines | `core.octave.resample` | lines | 1/6 octave bands | 1,000,000 | 251.7 ms | 1.18 |
