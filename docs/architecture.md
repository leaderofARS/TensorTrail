# Architecture

TensorTrail is split into a thin Python orchestration layer and a numerical Rust
core. The Python layer owns input provenance, policy configuration, reporting,
and backend selection. The Rust layer owns validation, reductions, and parallel
execution. The NumPy backend mirrors the policy for parity testing and
environments where the extension cannot be built.

## Package layout

```text
python/tensortrail/
  api.py                  public functions: compare, validate, metric helpers
  config.py               ComparisonConfig validation and metric names
  comparison/engine.py    normalize -> dispatch -> assemble report
  core/metadata.py        dtype/layout normalization and provenance
  core/result.py          immutable result dataclasses and summary rendering
  execution/              execution metadata, checkpoints, sessions, artifacts
  validation/             policy, alignment, validation engine, results, reporting
  backends/               BackendResult contract, NumPy and native adapters
  backends/pytorch_backend.py  lazy, explicit CPU tensor conversion
  reporting/              strict JSON and self-contained HTML writers
  exceptions.py           public exception hierarchy

native/tensortrail-core/src/
  lib.rs                  library entry points and PyO3 module registration
  tensor.rs               shape/stride arithmetic and dtype tags
  validation.rs           shared option and length validation
  error.rs                native error type
  metrics/                standalone and fused numerical kernels
  parallel/               deterministic chunking, cached Rayon pools
  bindings.rs             PyO3 functions and result objects
```

## Request flow

1. `tensortrail.compare(...)` validates the configuration and resolves the
   optional per-call metric override.
2. `engine.compare` normalizes both inputs independently. Each input becomes a
   contiguous, aligned, native-endian float32 or float64 buffer. The original
   dtype, device, shape, byte size, source, and conversion notes are retained.
3. `normalize_pair` verifies exact shape equality. If the working dtypes differ,
   both sides are upcast to float64 and the change is recorded.
4. The configured backend is dispatched explicitly. `rust` calls
   `tensortrail._core`; `numpy` calls the reference implementation. There is no
   fallback path.
5. The backend returns one `BackendResult`, a backend-neutral container with
   scalar metrics, counts, optional statistics, and execution metadata.
6. The engine selects only the metrics the caller requested, attaches tolerance
   and environment metadata, and constructs a frozen `ComparisonReport`.

The fused native kernel always computes the binary metrics in one traversal.
Requesting a subset changes only what is exposed in the report, not how the
kernel executes. `"statistics"` opts into Welford accumulation during the same
traversal.

## Execution validation flow

`tt.validate(reference, candidate, policy, alignment)` adds a layer above
`compare` without changing it:

1. Both sessions are checked for `completed` status.
2. The policy is resolved; a supplied alignment is checked against both
   execution IDs and the policy's checkpoint selection.
3. Checkpoints are aligned by explicit mapping first, then by unique name.
   Unmatched checkpoints remain explicit `missing_reference` /
   `missing_candidate` entries.
4. Each matched pair is dispatched to `tt.compare()` with the policy's
   `ComparisonConfig`.
5. Missing, shape-incompatible, dtype-rejected, invalid, and backend-failed
   cases become explicit outcomes; they never become passes.
6. Checkpoint outcomes are aggregated into `passed`, `failed`, or `incomplete`
   and assembled into an immutable `ValidationResult`.

A supplied alignment must cover every checkpoint in both sessions unless a
`checkpoint_selection` is configured. This prevents an incomplete alignment from
silently reducing the scope of validation. The engine does not import PyTorch or
inspect framework graphs; capture remains an explicit caller action.

## Native binding contract

The Python layer guarantees the following before calling the extension:

- exactly matching shapes;
- C-contiguous and aligned buffers;
- native byte order;
- dtype float32 or float64 only.

The Rust binding re-checks shapes and buffer contiguity and returns a Python
`ValueError` if the contract is violated. It never reinterprets a strided buffer
or copies memory silently. The Python adapter converts this into a
`TensorValidationError`/`BackendError` with context.

`release_gil=True` uses PyO3's `detach` to release the GIL around the numerical
work. Inputs are borrowed read-only for the duration; callers must not mutate
them concurrently. The flag and the observed native duration are recorded in
the report.

## Parallel execution

`map_chunks` splits an input into fixed-size chunks determined only by length
and `chunk_size`. Each chunk produces an independent accumulator. Partials are
merged in chunk order, not completion order, so scheduling cannot reorder
floating-point additions.

- Inputs smaller than `parallel_threshold` run sequentially.
- `threads=1` runs sequentially.
- A single chunk runs sequentially.
- Otherwise chunks run on a Rayon pool.

Pools are cached by requested thread count and reused across calls. `None` uses
the global Rayon pool. `ExecutionInfo` records whether the parallel path was
used, the worker count, the number of chunks, and the chunk size.

Two reproducibility guarantees follow from this design:

1. For a fixed chunk size and the parallel path, results are identical across
   worker counts and executions.
2. The sequential path may differ from a parallel run in the last few bits
   because the reduction tree differs. Changing `chunk_size` can also change
   rounding. Neither is presented as bit-exact reproducibility across different
   configurations.

## Backend contract

A backend must return `BackendResult` with:

- total and finite-pair counts;
- MAE, MSE, RMSE, relative mean/max, cosine similarity, maximum absolute error;
- finite and non-finite mismatch counts;
- per-side NaN/±Inf counts;
- optional per-side `TensorStatistics` when requested;
- execution metadata.

The engine checks that requested statistics are present and raises
`BackendError` if a backend violates the contract. Metric values may be `None`
only when mathematically undefined (no finite pairs). The NumPy backend reports
`parallel=False`, one thread, and one chunk.

## Optional PyTorch detection

`looks_like_torch_tensor` checks the type's module name. Ordinary NumPy inputs
never import torch. When a tensor is supplied, the adapter imports torch,
validates the device/layout/dtype, detaches autograd state, converts to NumPy,
and records each action. CUDA, MPS, meta, quantized, sparse, and complex tensors
are rejected rather than silently converted. This is deliberate: a report that
says `device="cpu"` must not hide a device transfer.

Execution sessions use the same detection only to describe a checkpoint's shape,
dtype, byte size, and device. Registering a CPU torch tensor does not convert it;
validation converts through the comparison adapter when a pair is compared.
Persistence requires an explicit NumPy payload, because storing a framework
object or converting it implicitly would violate the explicit-data-movement
rule.

## Reports

Comparison reports are immutable dataclasses. The JSON serializer converts
tuples to lists, emits a schema version, and encodes non-finite floats as strings
so the document remains valid under `json.loads`. HTML reports embed the escaped
JSON and use no external assets. Writes go through a temporary file in the
destination directory followed by an atomic replace.

Validation reports are a separate document type (`artifact_type:
"validation"`) and include both execution metadata objects, the policy, the full
alignment, per-checkpoint outcomes, embedded comparison reports, derived counts,
and the highest observed deviation. Filesystem artifacts add versioned
manifests, SHA-256 checksums, safe path validation, staged directory publication,
and pickle-free `.npy` loading. See
[execution_model.md](execution_model.md).

## Error model

All public exceptions derive from `TensorTrailError`:

- `TensorValidationError` for invalid values or configurations;
- `TensorShapeError` for shape mismatches;
- `UnsupportedDtypeError` for rejected dtypes;
- `BackendError` for unavailable or failed native execution;
- `ExecutionStateError` for illegal session transitions or validating a
  non-completed session;
- `CheckpointError` / `CheckpointNotFoundError` for registration and lookup
  failures;
- `AlignmentError` for invalid, ambiguous, or incomplete alignment;
- `ArtifactError` / `ArtifactIntegrityError` for missing, malformed, or
  tampered persistence.

Errors include the role (`reference`/`candidate`), offending dtype or shape, and
an actionable next step. Numerical policy never converts an error into a
plausible-looking number. During validation, comparison-time tensor/backend
errors are converted into explicit checkpoint outcomes where the engine can
continue; configuration and lifecycle errors raise before comparison.
