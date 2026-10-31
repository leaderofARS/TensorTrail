"""End-to-end Python benchmarks for TensorTrail.

Measures public API latency and throughput, including input normalization,
backend dispatch, native binding overhead, and report construction. Kernel-level
microbenchmarks live in ``native/tensortrail-core/benches/metrics.rs``.

Usage::

    python benchmarks/benchmark_python.py
    python benchmarks/benchmark_python.py --sizes 100000 1000000 --repeat 10
    python benchmarks/benchmark_python.py --json benchmarks/results/python.json

The script uses only NumPy and the standard library. ``psutil`` is used for CPU
metadata when installed.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tensortrail as tt

DEFAULT_SIZES = (1_000, 100_000, 1_000_000, 5_000_000)
DEFAULT_BACKENDS = ("rust", "numpy")


@dataclass
class Result:
    backend: str
    dtype: str
    elements: int
    metrics: str
    threads: int | None
    repeat: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    median_ns_per_element: float
    median_elements_per_second: float
    mae: float | None


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tensortrail": tt.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "native_available": tt.is_native_available(),
    }
    if tt.is_native_available():
        info["native"] = tt.native_info()
    try:
        import psutil

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["logical_cpus"] = psutil.cpu_count(logical=True)
        info["memory_gib"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        pass
    return info


def _data(size: int, dtype: np.dtype[Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=size).astype(dtype)
    candidate = (reference + rng.normal(scale=1e-4, size=size)).astype(dtype)
    return reference, candidate


def _time_compare(
    reference: np.ndarray,
    candidate: np.ndarray,
    config: tt.ComparisonConfig,
    *,
    warmup: int,
    repeat: int,
) -> tuple[list[float], tt.ComparisonReport]:
    last: tt.ComparisonReport | None = None
    for _ in range(warmup):
        last = tt.compare(reference, candidate, config=config)
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        last = tt.compare(reference, candidate, config=config)
        samples.append((time.perf_counter() - started) * 1e3)
    assert last is not None
    return samples, last


def _run_case(
    backend: str,
    dtype: np.dtype[Any],
    size: int,
    metrics: Sequence[str],
    threads: int | None,
    *,
    warmup: int,
    repeat: int,
    seed: int,
) -> Result:
    reference, candidate = _data(size, dtype, seed)
    chunk_size = max(1, min(32_768, size))
    config = tt.ComparisonConfig(
        backend=backend,
        metrics=tuple(metrics),
        threads=threads,
        chunk_size=chunk_size,
        parallel_threshold=32_768,
    )
    samples, report = _time_compare(
        reference,
        candidate,
        config,
        warmup=warmup,
        repeat=repeat,
    )
    median_ms = statistics.median(samples)
    return Result(
        backend=backend,
        dtype=np.dtype(dtype).name,
        elements=size,
        metrics="all" if tuple(metrics) == tt.DEFAULT_METRICS else "+".join(metrics),
        threads=threads,
        repeat=repeat,
        median_ms=median_ms,
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        median_ns_per_element=median_ms * 1e6 / size,
        median_elements_per_second=size / (median_ms / 1e3),
        mae=report.metrics.mae,
    )


def _print_table(results: list[Result]) -> None:
    header = (
        f"{'backend':<8} {'dtype':<8} {'elements':>10} {'metrics':<16} "
        f"{'threads':>7} {'median ms':>10} {'M elem/s':>10} {'ns/elem':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        threads = "auto" if result.threads is None else str(result.threads)
        print(
            f"{result.backend:<8} {result.dtype:<8} {result.elements:>10,} "
            f"{result.metrics:<16} {threads:>7} {result.median_ms:>10.3f} "
            f"{result.median_elements_per_second / 1e6:>10.2f} "
            f"{result.median_ns_per_element:>9.2f}"
        )


def _parse_sizes(values: Sequence[str]) -> tuple[int, ...]:
    sizes = tuple(int(value) for value in values)
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return sizes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=["rust", "numpy"],
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--threads", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    sizes = _parse_sizes(arguments.sizes)
    threads_to_test: list[int | None] = list(arguments.threads) if arguments.threads else [None]

    if "rust" in arguments.backends and not tt.is_native_available():
        parser.error("the rust backend is not available in this environment")

    environment = _environment()
    print("TensorTrail end-to-end benchmark")
    for key, value in environment.items():
        if key != "native":
            print(f"  {key}: {value}")
    if "native" in environment:
        print(f"  native: {environment['native']}")
    print()

    results: list[Result] = []
    for size in sizes:
        for dtype_name in arguments.dtypes:
            dtype = np.dtype(dtype_name)
            for backend in arguments.backends:
                for threads in threads_to_test:
                    metrics: tuple[str, ...] = tt.DEFAULT_METRICS
                    results.append(
                        _run_case(
                            backend,
                            dtype,
                            size,
                            metrics,
                            threads,
                            warmup=arguments.warmup,
                            repeat=arguments.repeat,
                            seed=arguments.seed,
                        )
                    )
                    if backend == "rust" and threads is None and size <= 1_000_000:
                        results.append(
                            _run_case(
                                backend,
                                dtype,
                                size,
                                ("mae",),
                                threads,
                                warmup=arguments.warmup,
                                repeat=arguments.repeat,
                                seed=arguments.seed,
                            )
                        )

    _print_table(results)

    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": environment,
            "results": [asdict(result) for result in results],
        }
        arguments.json.write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
