# Numerical methods and policy

This document is the contract for what TensorTrail computes. The Rust engine
and the NumPy reference backend implement the same policy; randomized parity
tests compare them. Where a policy cannot be exact in floating point, the
limitation is stated rather than hidden.

## Working precision

- Inputs are float32 or float64 after normalization.
- Both are widened to float64 for accumulation and results.
- Float16 and bfloat16 are converted to float32 first (their values are exactly
  representable in float32).
- Integers and bools are converted to float64. Int64/uint64 values above
  `2^53` may not be exactly representable in float64; this is recorded in
  `conversion_notes` as `integer_to_float64_may_lose_precision_above_2^53`.
- Mixed float32/float64 inputs are both upcast to float64.
- Outputs are Python floats (`float64`).

## Definitions

All sums run over **finite reference/candidate pairs** unless stated otherwise.
Let `n` be the number of finite pairs, `r_i` the reference value, `c_i` the
candidate value, and `epsilon = relative_epsilon`.

```text
MAE  = (1/n) * sum_i |r_i - c_i|
MSE  = (1/n) * sum_i (r_i - c_i)^2
RMSE = sqrt(MSE)
rel_i = |r_i - c_i| / max(|r_i|, epsilon)
relative_error.mean = (1/n) * sum_i rel_i
relative_error.max  = max_i rel_i
max_abs_error       = max_i |r_i - c_i|
```

`max_abs_error` is reported even when only a metric subset is requested; it is a
comparison-level diagnostic, not one of the selectable metric fields.

If `n == 0`, every one of these values is `None`. TensorTrail does not return
`0.0` for an empty or entirely non-finite comparison.

### Tolerance and status

For every finite pair:

```text
mismatch  <=>  |r_i - c_i| > atol + rtol * |r_i|
```

For every non-finite pair, the values match exactly when:

- both are NaN and `equal_nan=True`; or
- both are infinite with the same sign.

Every other non-finite pair is a non-finite pattern mismatch. The report status
is `"pass"` only when the sum of finite and non-finite mismatches is zero.
Status does not depend on which metrics were requested.

### Relative error epsilon

The denominator floor `max(|r|, epsilon)` prevents division by zero. A pair of
zeros contributes `0`. A zero reference with a nonzero candidate is measured
against `epsilon`, which can produce a large but finite relative error.

### Cosine similarity

For finite pairs:

```text
cos = dot(r, c) / sqrt(sum(r_i^2) * sum(c_i^2))
```

The result is clamped to `[-1, 1]` to absorb rounding. Policies:

- both vectors all zero: `1.0`;
- exactly one vector all zero: `0.0`;
- no finite pairs: `None`.

If the squared norms or their product overflow, underflow to zero, or become
non-finite, TensorTrail runs a scaled two-pass fallback. Each vector is divided
by its maximum absolute finite value before accumulating the dot product and
norms, keeping intermediate products bounded by one. This makes values such as
`1e308` and `1e-300` usable without changing the documented zero-vector policy.
The fallback costs extra traversals and is only triggered for extreme inputs.

## Non-finite values

NaN, +Inf, and -Inf are counted separately on the reference and candidate sides.
A pair is included in metric sums only when **both** values are finite.
Therefore one NaN does not turn an entire report into NaN.

- `counts.elements` counts all compared positions.
- `counts.finite_elements` counts positions where both values are finite.
- `counts.reference_nonfinite` / `candidate_nonfinite` count categories.
- `counts.finite_mismatches` counts tolerance failures among finite pairs.
- `counts.nonfinite_mismatches` counts mismatched non-finite patterns.
- `counts.mismatches` is their sum.

Statistics follow the same exclusion rule: non-finite values are counted but do
not contribute to min, max, mean, variance, or standard deviation.

## Statistics

For finite values only:

```text
min, max
mean     = (1/n) * sum_i x_i
variance = (1/n) * sum_i (x_i - mean)^2      # population variance
std_dev  = sqrt(variance)
```

The native engine uses Welford's online algorithm. Partial Welford states from
fixed chunks are merged with the Chan et al. parallel update. This avoids the
catastrophic cancellation of `sum(x^2) - sum(x)^2/n` for large offsets. The
NumPy reference uses a two-pass centered computation.

If no finite values exist, numeric fields are `None`. A single finite value has
variance and standard deviation `0.0`.

## Determinism

The native engine partitions input into chunks using only `len` and
`chunk_size`. Partial accumulators are merged in chunk order, never completion
order. Consequently:

- For a fixed chunk size and `threads >= 2` on a parallel-sized input, results
  are identical across worker counts and repeated runs.
- The sequential path (`threads=1`, input below `parallel_threshold`, or a
  single chunk) uses one chunk for the whole input and can differ from a
  parallel run in the last bits.
- Changing `chunk_size` changes chunk boundaries and therefore rounding.

This is deterministic execution, not a claim of bit-identical results across
different reduction layouts. Tests assert exact equality across worker counts
on the parallel path and approximate equality between sequential and parallel
paths.

## Input conversion

Normalization is explicit and every action appears in
`TensorInfo.conversion_notes`.

| Input | Working dtype | Note |
| --- | --- | --- |
| float32 + float32 | float32 | none |
| float64 + float64 | float64 | none |
| float32 + float64 | float64 | `upcast_to_float64_for_mixed_dtypes` |
| float16 | float32 | `float16_to_float32` |
| bfloat16 (from a dtype library such as `ml_dtypes`) | float32 | `bfloat16_to_float32` |
| bool | float64 | `bool_to_float64` |
| integer | float64 | `<dtype>_to_float64` |
| int64/uint64 | float64 | plus precision warning above 2^53 |
| non-native byte order | native | `non_native_byteorder_to_native` |
| non-contiguous / negative stride | contiguous | `non_contiguous_to_contiguous` |
| unaligned | aligned copy | `unaligned_to_aligned` |
| CPU torch half/bfloat16/float8 | float32 | `torch_<dtype>_to_float32` |
| CPU torch integer/bool | float64 | `torch_<dtype>_to_float64` |
| CPU torch non-contiguous | contiguous | `torch_non_contiguous_to_contiguous` |

Rejected rather than converted:

- complex64/complex128;
- object, unicode, bytes, datetime64, timedelta64;
- extended-precision floats (for example 80-bit/128-bit long double);
- masked arrays (fill or drop the mask explicitly);
- ragged or heterogeneous Python sequences;
- quantized, sparse, complex, CUDA/MPS/meta PyTorch tensors.

The native engine receives only contiguous, aligned, native-endian buffers. It
validates this again at the binding boundary.

NumPy has no built-in bfloat16 dtype. The normalization path recognizes a dtype
named `bfloat16` when a dtype library supplies one; the default test suite covers
bfloat16 through optional CPU PyTorch tensors, which use the torch-specific note
above.

## Overflow and underflow

- Sums and metrics may overflow to `Infinity` for extreme finite inputs, for
  example `|1e308 - (-1e308)|`. This is IEEE-754 behavior; the value is not
  silently clamped. The report serializer encodes it as `"Infinity"`.
- Infinite MAE/MSE/RMSE/relative error values always fail a finite tolerance
  check.
- Squaring a very small difference can underflow to `0.0`, so MSE/RMSE can
  report zero for nonzero differences below roughly `1e-162` in float64. The
  absolute difference, maximum absolute error, and relative error retain more
  information. This limitation is inherent to squared-error metrics in float64.
- Cosine avoids the corresponding norm under/overflow with the scaled fallback.
- Statistics can overflow for values near the float64 limit (variance is
  squared); the result is `Infinity` rather than a fabricated finite number.

## Cross-backend validation

The NumPy backend is the parity oracle. It uses NumPy reductions for metric
sums and a two-pass variance. The native backend uses ordered chunk merges and
Welford statistics. Different summation orders produce last-bit differences, so
tests use justified tolerances rather than demanding bit equality:

- Metrics on 100,003 random float32/float64 elements: `rel=1e-9`, `abs=1e-12`.
  A conservative bound for summing `n` values in float64 is on the order of
  `n * 2^-53`, about `1.1e-11` at `n=100,003`; the test margin is roughly two
  orders larger to cover cancellation and different reduction trees.
- Extreme-magnitude and hand-calculated cases: `rel=1e-10`, `abs=1e-14`.
- Statistics with a `1e6` offset over 50,003 values: `rel=1e-9` for variance.
  Welford and two-pass variance both avoid cancellation, but their operation
  order differs.
- `max_abs_error`, min, max, element counts, non-finite counts, mismatch counts,
  and status are compared exactly because they do not accumulate.

These margins are test tolerances, not accuracy guarantees for arbitrary
ill-conditioned data.

## Backend differences

- The NumPy backend always promotes working arrays to float64, even when both
  inputs are float32. The native backend keeps float32 inputs as float32 and
  widens each element to float64 during accumulation.
- The NumPy backend uses NumPy's pairwise reductions; the native backend uses
  fixed chunk merges.
- Both implement the same non-finite, empty, tolerance, relative-error, zero-
  vector, and statistic policies. Randomized parity tests cover each.
